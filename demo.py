"""nesting2d 演示: 固定输入, 真实计算, 离线运行。

演示内容:
1. 一个正常排样算例(含禁放格), 打印字符布局与最优性状态;
2. 固定放置(锁定)算例: 锁定一块已切位置, 优化器只重排剩余零件,
   并实际触发固定项冲突的直接拒绝;
3. 一个实际触发的输入拒绝(不连通零件 -> InvalidSpecError);
4. 一个实际触发的搜索预算耗尽(与证明无解/最优严格区分)。
"""

import time

from nesting2d import (
    Board,
    InvalidSpecError,
    LockedPlacement,
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


def demo_locked():
    print("=" * 56)
    print("算例 2: 固定放置(锁定) —— 已切位置锁住, 剩余零件重排")
    print("  7x5 板材; 零件0: 1x4 长条, 锁定竖放在 (5,1)")
    print("  零件1/2: 2x2 方块(自由); 零件3: 1x2 多米诺(自由)")
    board = Board(7, 5)
    parts = [
        Part([(0, 0), (1, 0), (2, 0), (3, 0)], rotations=(0, 90)),
        Part([(0, 0), (1, 0), (0, 1), (1, 1)], rotations=(0,)),
        Part([(0, 0), (1, 0), (0, 1), (1, 1)], rotations=(0,)),
        Part([(0, 0), (1, 0)], rotations=(0, 90)),
    ]
    locked = [LockedPlacement(0, 5, 1, 90)]
    t0 = time.perf_counter()
    res = pack(board, parts, locked=locked)
    dt = time.perf_counter() - t0
    print(f"状态: {res.status.value}  (耗时 {dt * 1000:.1f} ms, 节点 {res.nodes})")
    print(f"最小使用高度: {res.height}  旋转次数(仅自由件): "
          f"{res.rotation_count}  下界: {res.lower_bound}")
    print(f"布局坐标: {res.layout}")
    assert res.layout[0] == (5, 1, 90, False), "锁定件不得被移动"
    problems = verify_layout(board, parts, res.placements, res.height,
                             locked=locked)
    print(f"约束校验(含锁定一致性): {'通过' if not problems else problems}")
    print("字符布局(顶行在上; 零件0为锁定件):")
    print(render_layout(board, parts, res.placements, res.height))
    print("固定项之间冲突 -> 直接拒绝(不移开锁定件):")
    try:
        pack(board, parts, locked=[
            LockedPlacement(0, 5, 1, 90),
            LockedPlacement(1, 5, 3, 0),  # 与锁定长条边距不足
        ])
    except InvalidSpecError as e:
        print(f"已拒绝 -> InvalidSpecError: {e}")
    print("锁定旋转不合法 -> 直接拒绝:")
    try:
        pack(board, parts, locked=[LockedPlacement(1, 0, 0, 90)])
    except InvalidSpecError as e:
        print(f"已拒绝 -> InvalidSpecError: {e}")


def demo_rejection():
    print("=" * 56)
    print("算例 3: 非法输入的实际拒绝(不连通零件)")
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
    print("算例 4: 搜索预算耗尽(报告最优已知与下界, 不假装最优)")
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


def main():
    t0 = time.perf_counter()
    demo_normal()
    demo_locked()
    demo_rejection()
    demo_budget()
    print("=" * 56)
    print(f"演示总耗时: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
