"""固定输入演示：正常排样、证明不可行、非法输入拒绝。

运行：python demo.py   （纯标准库，离线，数秒内完成真实计算）
"""

from nesting import (
    Board,
    NestingError,
    Piece,
    STATUS_BUDGET_EXHAUSTED,
    STATUS_INFEASIBLE,
    STATUS_OPTIMAL,
    render,
    solve,
)

# 确定性固定零件库（仓库内代码生成，无外部数据）
RING = tuple((x, y) for x in range(3) for y in range(3) if (x, y) != (1, 1))
L4 = ((0, 0), (1, 0), (2, 0), (0, 1))
T4 = ((0, 0), (1, 0), (2, 0), (1, 1))
DOMINO = ((0, 0), (1, 0))
SQUARE = ((0, 0), (1, 0), (0, 1), (1, 1))


def show(title, board, pieces, result):
    print("=" * 60)
    print(title)
    print("-" * 60)
    print(render(result, board, pieces))
    print()


def main():
    # 用例 1：正常排样（含带孔零件、重复形、禁放格、旋转/镜像策略）
    board = Board(8, 6, forbidden={(4, 0)})
    pieces = [
        Piece(RING, allowed_rotations=(0, 90, 180, 270)),          # A 带孔环
        Piece(L4, allowed_rotations=(0, 90, 180, 270), allow_mirror=True),  # B
        Piece(T4, allowed_rotations=(0, 90, 180, 270)),            # C
        Piece(DOMINO, allowed_rotations=(0, 90)),                  # D 重复形
        Piece(DOMINO, allowed_rotations=(0, 90)),                  # E 重复形
        Piece(SQUARE, allowed_rotations=(0,)),                     # F 禁旋转
    ]
    res = solve(board, pieces)
    assert res.status == STATUS_OPTIMAL, res
    show("用例 1：6 件排样（A 带孔环 / D、E 重复形 / F 禁旋转 / 禁放格 (4,0)）",
         board, pieces, res)

    # 用例 2：证明不可行（与预算耗尽严格区分）
    board2 = Board(3, 3)
    pieces2 = [Piece(SQUARE, allowed_rotations=(0,)),
               Piece(SQUARE, allowed_rotations=(0,))]
    res2 = solve(board2, pieces2)
    assert res2.status == STATUS_INFEASIBLE, res2
    show("用例 2：两个 2x2 方块放入 3x3 板（一格边距下证明不可行）",
         board2, pieces2, res2)

    # 用例 3：实际触发的非法输入拒绝
    print("=" * 60)
    print("用例 3：非法输入拒绝（不连通的零件）")
    print("-" * 60)
    try:
        Piece(cells=((0, 0), (2, 2)))
    except NestingError as exc:
        print(f"已拒绝并定位：{type(exc).__name__}: {exc}")
    print()

    # 用例 4：预算耗尽与“证明不可行”是不同状态
    print("=" * 60)
    print("用例 4：显式搜索预算（budget=1 必然耗尽）")
    print("-" * 60)
    res4 = solve(Board(2, 2), [Piece(((0, 0),))], budget=1)
    assert res4.status == STATUS_BUDGET_EXHAUSTED, res4
    print(render(res4))
    print()


if __name__ == "__main__":
    main()
