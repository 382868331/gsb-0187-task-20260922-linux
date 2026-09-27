"""nesting2d 演示: 固定输入, 真实计算, 离线运行。

演示内容:
1. 一个正常排样算例(含禁放格), 打印字符布局与最优性状态;
2. 一个实际触发的输入拒绝(不连通零件 -> InvalidSpecError);
3. 一个实际触发的搜索预算耗尽(与证明无解/最优严格区分)。
"""

import time

from nesting2d import (
    Board,
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


def demo_damage_wall():
    print("=" * 56)
    print("算例 4: 损坏柱把板材隔成多块(损坏区连边界)")
    # 损坏柱 x=2 贯穿 5 行, 把 5x5 板切成两个宽 2 的连通块;
    # 两个多米诺只能各占一块, 高度 1 横放即可。
    wall = [(2, y) for y in range(5)]
    board = Board(5, 5, forbidden=wall)
    domino = Part([(0, 0), (1, 0)], rotations=(0, 90))
    res = pack(board, [domino, domino])
    print(f"2 个多米诺 -> 状态: {res.status.value}, 高度: {res.height}, "
          f"旋转次数: {res.rotation_count}")
    print(render_layout(board, [domino, domino], res.placements, res.height))
    assert res.status is Status.OPTIMAL and res.height == 1
    # 每个宽 2 的块(高 5 行)含一格边距最多横放 3 个多米诺(y=0,2,4),
    # 两块共 6 个; 第 7 个无处可放: 给足预算后被证明无解。
    seven = [domino] * 7
    full = pack(board, seven)
    print(f"7 个多米诺(给足预算) -> 状态: {full.status.value} (已证明无解)")
    assert full.status is Status.INFEASIBLE
    # 同一实例预算=1: 搜索未结束, 只能报告预算耗尽, 不能说无解
    tiny = pack(board, seven, budget=1)
    print(f"7 个多米诺(budget=1) -> 状态: {tiny.status.value} (搜索未结束, "
          f"下界 {tiny.lower_bound})")
    assert tiny.status is Status.BUDGET_EXHAUSTED


def main():
    t0 = time.perf_counter()
    demo_normal()
    demo_rejection()
    demo_budget()
    demo_damage_wall()
    print("=" * 56)
    print(f"演示总耗时: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
