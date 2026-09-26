"""nesting2d 迭代演示: 整刀(guillotine)可切性检查与切割树生成。

固定输入, 真实计算, 离线运行(远小于 90 秒)。演示内容:
1. 一个可整刀加工的布局: 先 pack 出排样, 再生成最优切割树,
   打印每刀作用的子板范围;
2. 一个"风车形"布局: 不存在第一刀, 证明不可切并给出阻挡见证;
3. 预算耗尽(BUDGET_EXHAUSTED)与确定不可切(NOT_CUTTABLE)的区分。
"""

import time

from nesting2d import (
    Board,
    GuillotineStatus,
    Part,
    Placement,
    guillotine_cuts,
    pack,
    render_layout,
)


def print_tree(node, parts, indent=0):
    pad = "  " * indent
    x0, y0, x1, y1 = node.rect
    if node.is_leaf:
        if node.is_waste:
            print(f"{pad}废料叶 子板=({x0},{y0})-({x1},{y1})")
        else:
            print(f"{pad}零件叶 子板=({x0},{y0})-({x1},{y1}) "
                  f"零件={node.part_index}")
        return
    cut = node.cut
    direction = "竖刀" if cut.axis == "v" else "横刀"
    line = f"x={cut.position}" if cut.axis == "v" else f"y={cut.position}"
    print(f"{pad}{direction} {line} 子板=({x0},{y0})-({x1},{y1}) "
          f"切长={cut.length}")
    for child in node.children:
        print_tree(child, parts, indent + 1)


def demo_cuttable():
    print("=" * 60)
    print("算例 1: 可整刀加工的布局(pack 排样 + 切割树生成)")
    board = Board(8, 6)
    parts = [
        Part([(0, 0), (1, 0)], rotations=(0,)),          # 0: 1x2 横条
        Part([(0, 0), (0, 1)], rotations=(0,)),          # 1: 1x2 竖条
        Part([(0, 0), (1, 0), (0, 1), (1, 1)], rotations=(0,)),  # 2: 2x2
        Part([(0, 0)], rotations=(0,)),                  # 3: 单格
    ]
    res = pack(board, parts)
    print(f"排样状态: {res.status.value}, 使用高度: {res.height}")
    print(render_layout(board, parts, res.placements, res.height))
    gres = guillotine_cuts(board, parts, res.placements, res.height)
    print(f"整刀结论: {gres.status.value}, 刀数: {gres.cut_count}, "
          f"总切长: {gres.total_cut_length}, 展开状态数: {gres.nodes}")
    print("切割树(每刀标注作用的子板范围):")
    print_tree(gres.tree, parts)


def demo_windmill():
    print("=" * 60)
    print("算例 2: 风车形布局 —— 不存在第一刀, 证明不可整刀加工")
    board = Board(9, 9)
    parts = [
        Part([(x, 0) for x in range(6)], rotations=(0,)),   # A 横条(上)
        Part([(0, y) for y in range(6)], rotations=(0,)),   # B 竖条(右)
        Part([(x, 0) for x in range(6)], rotations=(0,)),   # C 横条(下)
        Part([(0, y) for y in range(7)], rotations=(0,)),   # D 竖条(左)
    ]
    placements = (
        Placement(0, 0, 8, 0, False),
        Placement(1, 8, 3, 0, False),
        Placement(2, 3, 0, 0, False),
        Placement(3, 0, 0, 0, False),
    )
    print(render_layout(board, parts, placements, 9))
    gres = guillotine_cuts(board, parts, placements, 9)
    print(f"整刀结论: {gres.status.value} (展开状态数: {gres.nodes})")
    assert gres.status is GuillotineStatus.NOT_CUTTABLE
    # 阻挡见证: 逐条候选第一刀给出挡住它的零件
    from nesting2d import oriented_cells

    boxes = []
    for pl in placements:
        shape = oriented_cells(
            parts[pl.part_index].cells, pl.rotation_degrees, pl.mirrored
        )
        xs = [x + pl.x for x, _ in shape]
        ys = [y + pl.y for _, y in shape]
        boxes.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1))
    print("阻挡见证(每条候选第一刀都被某零件拦住):")
    labels = "ABCD"
    for axis, rng in (("v", range(1, 9)), ("h", range(1, 9))):
        for c in rng:
            hit = [labels[i] for i, b in enumerate(boxes)
                   if (b[0] < c < b[2]) if axis == "v"]
            if axis == "h":
                hit = [labels[i] for i, b in enumerate(boxes)
                       if b[1] < c < b[3]]
            line = f"x={c}" if axis == "v" else f"y={c}"
            print(f"  候选{'竖刀' if axis == 'v' else '横刀'} {line}: "
                  f"被零件 {','.join(hit)} 阻挡")


def demo_budget():
    print("=" * 60)
    print("算例 3: 预算耗尽与确定不可切严格区分")
    board = Board(5, 5)
    parts = [Part([(0, 0)], rotations=(0,)) for _ in range(4)]
    placements = (
        Placement(0, 0, 0, 0, False),
        Placement(1, 4, 0, 0, False),
        Placement(2, 0, 4, 0, False),
        Placement(3, 4, 4, 0, False),
    )
    tight = guillotine_cuts(board, parts, placements, 5, budget=1)
    print(f"预算=1   -> {tight.status.value} (未证明可切也未证明不可切, "
          f"展开状态数: {tight.nodes})")
    full = guillotine_cuts(board, parts, placements, 5)
    print(f"默认预算 -> {full.status.value}, 刀数: {full.cut_count}, "
          f"总切长: {full.total_cut_length}")
    assert tight.status is GuillotineStatus.BUDGET_EXHAUSTED
    assert full.status is GuillotineStatus.CUTTABLE


def main():
    t0 = time.perf_counter()
    demo_cuttable()
    demo_windmill()
    demo_budget()
    print("=" * 60)
    print(f"演示总耗时: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()

