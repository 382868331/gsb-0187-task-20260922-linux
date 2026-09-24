"""nesting2d 演示: 固定输入, 真实计算, 离线运行。

演示内容:
1. 一个正常排样算例(含禁放格), 打印字符布局与最优性状态;
2. 一个实际触发的输入拒绝(不连通零件 -> InvalidSpecError);
3. 一个实际触发的搜索预算耗尽(与证明无解/最优严格区分);
4. 切割段生成: 共边只切一刀、部分重合切分、孔洞内嵌件、切缝补偿,
   以及实际触发的非法输入拒绝。
"""

import time
from fractions import Fraction

from nesting2d import (
    Board,
    InvalidSpecError,
    Part,
    Placement,
    PolyPart,
    PolyPlacement,
    Status,
    cut_segments,
    cut_segments_for_layout,
    pack,
    render_layout,
    verify_layout,
)


def demo_normal():
    print("=" * 56)
    print("算例 1: 5x5 板材, 禁放格 (4,3), 3 个零件")
    print("  零件0: 1x4 长条 (允许 0/90 度旋转)")
    print("  零件1: 2x2 方块 (不允许旋转)")
    print("  零件2: 2x2 方块 (不允许旋转)")
    board = Board(5, 5, forbidden=[(4, 3)])
    parts = [
        Part([(0, 0), (1, 0), (2, 0), (3, 0)], rotations=(0, 90)),
        Part([(0, 0), (1, 0), (0, 1), (1, 1)], rotations=(0,)),
        Part([(0, 0), (1, 0), (0, 1), (1, 1)], rotations=(0,)),
    ]
    t0 = time.perf_counter()
    res = pack(board, parts)
    dt = time.perf_counter() - t0
    print(f"状态: {res.status.value}  (耗时 {dt * 1000:.1f} ms, 节点 {res.nodes})")
    print(f"最小使用高度: {res.height}  旋转次数: {res.rotation_count}  "
          f"下界: {res.lower_bound}")
    print(f"布局坐标: {res.layout}")
    problems = verify_layout(board, parts, res.placements, res.height)
    print(f"约束校验: {'通过' if not problems else problems}")
    print("字符布局(顶行在上, '#'=禁放格, '.'=空白):")
    print(render_layout(board, parts, res.placements, res.height))


def demo_rejection():
    print("=" * 56)
    print("算例 2: 非法输入的实际拒绝(不连通零件)")
    try:
        Part([(0, 0), (2, 2)])  # 两个格子不相邻: 不连通
    except InvalidSpecError as e:
        print(f"已拒绝 -> InvalidSpecError: {e}")
    try:
        Board(3, 3, forbidden=[(9, 0)])  # 禁放格越界
    except InvalidSpecError as e:
        print(f"已拒绝 -> InvalidSpecError: {e}")
    try:
        Board(True, 3)  # bool 不是合法整数参数
    except TypeError as e:
        print(f"已拒绝 -> TypeError: {e}")


def demo_budget():
    print("=" * 56)
    print("算例 3: 搜索预算耗尽(报告最优已知与下界, 不假装最优)")
    board = Board(5, 5)
    parts = [
        Part([(0, 0), (1, 0)], rotations=(0, 90)),
        Part([(0, 0), (1, 0)], rotations=(0, 90)),
        Part([(0, 0), (1, 0)], rotations=(0, 90)),
    ]
    res = pack(board, parts, budget=1)
    print(f"预算=1   -> 状态: {res.status.value}, 最优已知高度: {res.height}, "
          f"下界: {res.lower_bound}, 节点: {res.nodes}")
    full = pack(board, parts)
    print(f"默认预算 -> 状态: {full.status.value}, 最优高度: {full.height}, "
          f"旋转次数: {full.rotation_count}, 节点: {full.nodes}")
    assert res.status is Status.BUDGET_EXHAUSTED
    assert full.status is Status.OPTIMAL


def _fmt_f(x):
    return str(x)


def demo_cuts():
    print("=" * 56)
    print("算例 4: 切割段生成(有理坐标直边, 精确 Fraction 计算)")
    # 布局(俯视):
    #   A = 4x1 横条, 占 [0,4] x [0,1]
    #   B = 2x2 方块, 占 [1,3] x [1,3], 底边与 A 顶边部分重合(T 形)
    #   C = 1x1 方块, 占 [4,5] x [0,1], 与 A 整边贴合(整段共边)
    A = PolyPart([(0, 0), (4, 0), (4, 1), (0, 1)])
    B = PolyPart([(0, 0), (2, 0), (2, 2), (0, 2)])
    C = PolyPart([(0, 0), (1, 0), (1, 1), (0, 1)])
    parts = [A, B, C]
    placements = [
        PolyPlacement(0, 0, 0),
        PolyPlacement(1, 1, 1),
        PolyPlacement(2, 4, 0),
    ]
    plan = cut_segments(parts, placements)
    print(f"原始轮廓边 {plan.raw_edge_count} 条 -> 切割段 {len(plan)} 段, "
          f"其中共边一刀 {plan.shared_count} 段")
    print("全部切割段(p1 -> p2, 涉及零件):")
    for s in plan.segments:
        tag = "  [共边, 只切一刀]" if s.shared else ""
        print(f"  ({_fmt_f(s.p1[0])},{_fmt_f(s.p1[1])}) -> "
              f"({_fmt_f(s.p2[0])},{_fmt_f(s.p2[1])}) 零件{s.parts}{tag}")
    # 手算核对(不调用被测实现):
    #  y=1 处必须分成 3 段: A-only / A+B 共享 / A-only
    y1 = [(s.p1[0], s.p2[0], s.parts)
          for s in plan.segments
          if s.p1[1] == 1 and s.p2[1] == 1 and s.p2[0] <= 4]
    assert y1 == [(Fraction(0), Fraction(1), (0,)),
                  (Fraction(1), Fraction(3), (0, 1)),
                  (Fraction(3), Fraction(4), (0,))], y1
    #  x=4 处 A 与 C 整边重合 -> 一整段共边
    assert any(s.shared and s.p1 == (4, 0) and s.p2 == (4, 1)
               and s.parts == (0, 2) for s in plan.segments)
    print("  手算核对通过: y=1 部分重合未被整条去重; x=4 整边只切一刀")

    print("-" * 56)
    print("切缝补偿不同: A、C 同一共边但 kerf 不同 -> 不合并")
    plan2 = cut_segments(
        [A, C],
        [PolyPlacement(0, 0, 0), PolyPlacement(1, 4, 0)],
        kerfs={0: Fraction(1, 10), 1: Fraction(1, 4)},
    )
    x4 = [s for s in plan2.segments
          if s.p1[0] == 4 and s.p2[0] == 4]
    for s in x4:
        print(f"  x=4 竖边 ({_fmt_f(s.p1[0])},{_fmt_f(s.p1[1])}) -> "
              f"({_fmt_f(s.p2[0])},{_fmt_f(s.p2[1])}) 零件{s.parts} "
              f"kerf={_fmt_f(s.kerf)} shared={s.shared}")
    assert len(x4) == 2 and all(not s.shared for s in x4)

    print("-" * 56)
    print("内外轮廓: 方块嵌入带孔零件的孔洞中, 孔边 4 条全部共边")
    host = PolyPart(
        [(0, 0), (5, 0), (5, 5), (0, 5)],
        holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
    )
    insert = PolyPart([(0, 0), (2, 0), (2, 2), (0, 2)])
    plan3 = cut_segments(
        [host, insert],
        [PolyPlacement(0, 0, 0), PolyPlacement(1, 1, 1)],
    )
    print(f"  共边段数 = {plan3.shared_count} (期望 4, 即四条孔边)")
    assert plan3.shared_count == 4


def demo_cuts_from_layout():
    print("=" * 56)
    print("算例 5: 从网格 pack 结果直接生成切割段")
    board = Board(3, 2)
    parts = [Part([(0, 0), (1, 0)], rotations=(0, 90)) for _ in range(2)]
    # 手工构造贴边放置(共边切割允许贴边; pack 默认保留一格边距):
    placements = [
        Placement(0, 0, 0, 90, False),
        Placement(1, 1, 0, 90, False),
    ]
    plan = cut_segments_for_layout(board, parts, placements, 2)
    print(f"  两个 1x2 竖条贴边: {plan.raw_edge_count} 条单位轮廓边 -> "
          f"{len(plan)} 段, 共边 {plan.shared_count} 段")
    shared = [s for s in plan.segments if s.shared][0]
    print(f"  共边: ({_fmt_f(shared.p1[0])},{_fmt_f(shared.p1[1])}) -> "
          f"({_fmt_f(shared.p2[0])},{_fmt_f(shared.p2[1])}) 零件{shared.parts}")
    assert shared.p1 == (1, 0) and shared.p2 == (1, 2)

    print("-" * 56)
    print("实际触发的切割输入拒绝:")
    sq = PolyPart([(0, 0), (2, 0), (2, 2), (0, 2)])
    try:
        cut_segments([sq, sq],
                     [PolyPlacement(0, 0, 0), PolyPlacement(1, 1, 0)])
    except InvalidSpecError as e:
        print(f"  区域重叠 -> InvalidSpecError: {e}")
    try:
        cut_segments([sq], [PolyPlacement(0, 0, 0)], kerf=0.5)
    except TypeError as e:
        print(f"  float 切缝宽 -> TypeError: {e}")
    try:
        cut_segments([sq], [PolyPlacement(0, 0, 0)], kerf=-1)
    except InvalidSpecError as e:
        print(f"  负切缝宽 -> InvalidSpecError: {e}")


def main():
    t0 = time.perf_counter()
    demo_normal()
    demo_rejection()
    demo_budget()
    demo_cuts()
    demo_cuts_from_layout()
    print("=" * 56)
    print(f"演示总耗时: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
