"""nesting2d 演示: 固定输入, 真实计算, 离线运行。

演示内容:
1. 一个正常排样算例(含禁放格), 打印字符布局与最优性状态;
2. 一个实际触发的输入拒绝(不连通零件 -> InvalidSpecError);
3. 一个实际触发的搜索预算耗尽(与证明无解/最优严格区分);
4. 固定放置约束: 两道锁定竖墙夹出窄洞, 只排自由零件;
   以及一个锁定旋转不合法的实际拒绝。
"""

import time

from nesting2d import (
    Board,
    FixedPlacement,
    InvalidSpecError,
    Part,
    Status,
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


def demo_fixed():
    print("=" * 56)
    print("算例 4: 固定放置约束(两道锁定竖墙夹出窄洞, 只排自由件)")
    board = Board(5, 3)
    vdom = lambda: Part([(0, 0), (0, 1)], rotations=(0,))
    parts = [vdom(), vdom(), Part([(0, 0)])]
    fixed = [FixedPlacement(0, 0, 0, 0), FixedPlacement(1, 4, 0, 0)]
    print("  零件0: 竖直 2 格, 锁定于 (0,0)")
    print("  零件1: 竖直 2 格, 锁定于 (4,0)")
    print("  零件2: 单格, 自由排样(只能落进两墙一格膨胀后的窄洞)")
    res = pack(board, parts, fixed=fixed)
    print(f"状态: {res.status.value}  高度: {res.height}  "
          f"可移动件旋转次数: {res.rotation_count}  下界: {res.lower_bound}")
    print(f"布局坐标(零件0/1 保持锁定): {res.layout}")
    problems = verify_layout(board, parts, res.placements, res.height, fixed=fixed)
    print(f"约束+锁定校验: {'通过' if not problems else problems}")
    print(render_layout(board, parts, res.placements, res.height))

    print("-" * 56)
    print("锁定旋转不合法 -> 搜索前直接拒绝(不会移开锁定件另给解):")
    try:
        pack(
            Board(4, 4),
            [Part([(0, 0), (1, 0)], rotations=(0,))],
            fixed=[FixedPlacement(0, 0, 0, 90)],
        )
    except InvalidSpecError as e:
        print(f"已拒绝 -> InvalidSpecError: {e}")


def main():
    t0 = time.perf_counter()
    demo_normal()
    demo_rejection()
    demo_budget()
    demo_fixed()
    print("=" * 56)
    print(f"演示总耗时: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
