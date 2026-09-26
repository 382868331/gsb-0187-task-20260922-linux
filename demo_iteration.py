"""nesting2d 整刀切割迭代演示: 固定输入, 真实计算, 离线运行。

演示内容:
1. 一个可整刀加工的布局: 打印每刀作用的子板范围与切割树;
2. 一个"风车形"无第一刀的布局: 打印每条候选切线的阻挡见证;
3. 预算耗尽(BUDGET_EXHAUSTED)与确定不可切(NOT_GUILLOTINEABLE)的区别。
"""

import time

from nesting2d import (
    Board,
    CutBranch,
    CutLeaf,
    GuillotineStatus,
    Part,
    Placement,
    guillotine_cut,
    render_layout,
)


def pl(i, x, y):
    return Placement(i, x, y, 0, False)


def print_tree(node, indent="  "):
    if isinstance(node, CutLeaf):
        what = "废料" if node.part_index is None else f"零件 {node.part_index}"
        print(f"{indent}叶 {node.rect}: {what}")
    else:
        c = node.cut
        direction = "竖切 x" if c.axis == "v" else "横切 y"
        print(f"{indent}刀 {direction}={c.coord} (子板 {c.rect}, 长 {c.length})")
        print_tree(node.lower, indent + "  ")
        print_tree(node.upper, indent + "  ")


def demo_cuttable():
    print("=" * 60)
    print("算例 1: 6x3 板材, 3 件(方块 + L + 竖多米诺), 可整刀加工")
    board = Board(6, 3)
    parts = [
        Part([(0, 0), (1, 0), (0, 1), (1, 1)]),   # 零件0: 2x2 方块 @ (0,0)
        Part([(0, 0), (1, 0), (1, 1)]),           # 零件1: L 三连块 @ (3,0)
        Part([(0, 0), (0, 1)]),                   # 零件2: 竖多米诺 @ (5,1)
    ]
    placements = (pl(0, 0, 0), pl(1, 3, 0), pl(2, 5, 1))
    print("布局(顶行在上):")
    print(render_layout(board, parts, placements, board.height))
    t0 = time.perf_counter()
    res = guillotine_cut(board, parts, placements)
    dt = time.perf_counter() - t0
    print(f"状态: {res.status.value}  (耗时 {dt * 1000:.1f} ms, 子板节点 {res.nodes})")
    assert res.status is GuillotineStatus.CUTTABLE
    print(f"刀数: {len(res.cuts)}  总切长: {res.total_length}")
    for i, cut in enumerate(res.cuts):
        direction = "竖切 x" if cut.axis == "v" else "横切 y"
        print(f"  刀{i}: {direction}={cut.coord}  作用子板 {cut.rect}  切长 {cut.length}")
    print("切割树:")
    print_tree(res.tree)


def demo_windmill():
    print("=" * 60)
    print("算例 2: 4x4 风车形(5 件铺满整板), 无第一刀 -> 确定不可整刀")
    board = Board(4, 4)
    parts = [
        Part([(0, 0), (0, 1), (1, 1)]),              # 0: 左上 L
        Part([(0, 1), (1, 1), (1, 0)]),              # 1: 右上 L
        Part([(1, 0), (1, 1), (0, 0)]),              # 2: 右下 L
        Part([(0, 0), (0, 1), (1, 0)]),              # 3: 左下 L
        Part([(0, 0), (1, 0), (0, 1), (1, 1)]),      # 4: 中心方块
    ]
    placements = (pl(0, 0, 2), pl(1, 2, 2), pl(2, 2, 0), pl(3, 0, 0), pl(4, 1, 1))
    print("布局(顶行在上):")
    print(render_layout(board, parts, placements, board.height))
    res = guillotine_cut(board, parts, placements)
    print(f"状态: {res.status.value}  (子板节点 {res.nodes})")
    assert res.status is GuillotineStatus.NOT_GUILLOTINEABLE
    print("阻挡见证(根子板每条候选切线都被零件内部挡住):")
    for b in res.blocking:
        direction = "竖切 x" if b.axis == "v" else "横切 y"
        print(f"  {direction}={b.coord}: 被零件 {list(b.parts)} 挡住")


def demo_budget_vs_infeasible():
    print("=" * 60)
    print("算例 3: 预算耗尽 与 确定不可切 严格区分")
    board = Board(6, 3)
    parts = [
        Part([(0, 0), (1, 0), (0, 1), (1, 1)]),
        Part([(0, 0), (1, 0), (1, 1)]),
        Part([(0, 0), (0, 1)]),
    ]
    placements = (pl(0, 0, 0), pl(1, 3, 0), pl(2, 5, 1))
    tight = guillotine_cut(board, parts, placements, budget=1)
    full = guillotine_cut(board, parts, placements)
    print(f"预算=1    -> {tight.status.value} (不下结论, cuts={tight.cuts})")
    print(f"默认预算  -> {full.status.value} (刀数 {len(full.cuts)}, 总切长 {full.total_length})")
    assert tight.status is GuillotineStatus.BUDGET_EXHAUSTED
    assert full.status is GuillotineStatus.CUTTABLE


def main():
    t0 = time.perf_counter()
    demo_cuttable()
    demo_windmill()
    demo_budget_vs_infeasible()
    print("=" * 60)
    print(f"演示总耗时: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()

