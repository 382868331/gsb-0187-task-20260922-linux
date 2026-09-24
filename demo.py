"""nesting2d 演示: 固定输入, 真实计算, 离线运行。

演示内容:
1. 一个正常排样算例(含禁放格), 打印字符布局与最优性状态;
2. 一个实际触发的输入拒绝(不连通零件 -> InvalidSpecError);
3. 一个实际触发的搜索预算耗尽(与证明无解/最优严格区分);
4. 切割段生成: 共边只切一次、部分重合拆分、内外轮廓(孔)、
   切缝补偿不同切两次, 以及实际触发的非法布局拒绝。
"""

import time
from fractions import Fraction

from nesting2d import (
    Board,
    InvalidSpecError,
    Part,
    Placement,
    Status,
    generate_cut_segments,
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


def demo_cutting():
    print("=" * 56)
    print("算例 4: 切割段生成 — 共边只切一次, 部分重合不错误去重")
    print("  件0: 横多米诺锚 (0,0); 件1: 单格锚 (1,1); 件2: 单格锚 (2,0)")
    board = Board(5, 3)
    parts = [
        Part([(0, 0), (1, 0)], rotations=(0, 90)),
        Part([(0, 0)]),
        Part([(0, 0)]),
    ]
    placements = [
        Placement(0, 0, 0, 0, False),
        Placement(1, 1, 1, 0, False),
        Placement(2, 2, 0, 0, False),
    ]
    plan = generate_cut_segments(board, parts, placements)
    for s in plan.segments:
        owner = ",".join(str(i) for i in sorted(s.part_ids))
        kind = "水平" if s.y1 == s.y2 else "竖直"
        print(f"  [{kind}] ({s.x1},{s.y1})->({s.x2},{s.y2}) 长 {s.length} "
              f"零件 {{{owner}}} 轮廓 {s.contours}")
    # 手算: 周长和 6+4+4=14; 件0/件2 完全重合竖边 1 + 件0/件1 部分重合水平边 1 => 12
    print(f"切割段数: {len(plan)}  实际切割总长: {plan.total_length} (手算 12)")
    assert plan.total_length == 12
    line_y1 = sorted((s.x1, s.x2, tuple(sorted(s.part_ids)))
                     for s in plan.segments if s.y1 == s.y2 == 1)
    print("  y=1 直线拆分:", line_y1,
          "— [0,1] 只属件0, [1,2] 才是件0/件1 的部分共边, [2,3] 只属件2")
    assert line_y1 == [(0, 1, (0,)), (1, 2, (0, 1)), (2, 3, (2,))]
    vert = [s for s in plan.segments if s.x1 == s.x2 == 2
            and s.part_ids == frozenset({0, 2})]
    assert len(vert) == 1 and vert[0].length == 1

    print("-" * 56)
    print("算例 5: 内外轮廓 — 3x3 环与填入孔内的单格")
    board2 = Board(5, 5)
    ring = Part(
        [(x, y) for x in range(3) for y in range(3) if (x, y) != (1, 1)],
        rotations=(0,),
    )
    parts2 = [ring, Part([(0, 0)])]
    placements2 = [Placement(0, 1, 1, 0, False), Placement(1, 2, 2, 0, False)]
    plan2 = generate_cut_segments(board2, parts2, placements2)
    shared = [s for s in plan2.segments if s.part_ids == frozenset({0, 1})]
    print(f"  共边 {len(shared)} 条, 每条 contours={shared[0].contours}, "
          f"聚合角色 contour={shared[0].contour!r}")
    # 环周长 16 + 单格 4 - 共边 4 = 16
    print(f"  实际切割总长: {plan2.total_length} (手算 16; 共边虽一内一外仍只切一次)")
    assert plan2.total_length == 16 and len(shared) == 4

    print("-" * 56)
    print("算例 6: 切缝补偿不同 — 同一几何边按补偿切两次")
    plan3 = generate_cut_segments(
        board, parts, placements, kerf_width=[0, 1, 0]
    )
    twins = sorted(
        (s.kerf_width, tuple(sorted(s.part_ids)))
        for s in plan3.segments
        if (s.x1, s.y1, s.x2, s.y2) == (1, 1, 2, 1)
    )
    print("  共边 [1,2]@y=1 上的切割:", twins,
          "— 件0 补偿 0、件1 补偿 1, 不合并")
    assert twins == [(Fraction(0), (0,)), (Fraction(1), (1,))]
    print(f"  实际切割总长: {plan3.total_length} (比同补偿多切 1)")
    assert plan3.total_length == 13


def demo_cutting_error():
    print("=" * 56)
    print("算例 7: 切割输入的实际拒绝(失败不留部分结果)")
    board = Board(4, 4)
    parts = [Part([(0, 0)]), Part([(0, 0)])]
    try:
        generate_cut_segments(
            board, parts,
            [Placement(0, 0, 0, 0, False), Placement(1, 0, 0, 0, False)],
        )  # 两个零件落在同一格: 重叠
    except InvalidSpecError as e:
        print(f"已拒绝 -> InvalidSpecError: {e}")
    try:
        generate_cut_segments(
            board, parts,
            [Placement(0, 0, 0, 0, False), Placement(1, 1, 0, 0, False)],
            kerf_width=0.25,
        )  # float 切缝补偿
    except TypeError as e:
        print(f"已拒绝 -> TypeError: {e}")
    # 同一批对象修正后立即可用: 证明失败没有留下半更新状态
    plan = generate_cut_segments(
        board, parts,
        [Placement(0, 0, 0, 0, False), Placement(1, 1, 0, 0, False)],
    )
    print(f"修正调用 -> {len(plan)} 条段, 总长 {plan.total_length}(函数式, 无副作用)")
    assert plan.total_length == 7


def main():
    t0 = time.perf_counter()
    demo_normal()
    demo_rejection()
    demo_budget()
    demo_cutting()
    demo_cutting_error()
    print("=" * 56)
    print(f"演示总耗时: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
