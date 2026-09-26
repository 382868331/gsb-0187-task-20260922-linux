"""整刀切割的独立暴力参考实现(验收用)。

与被测核心 nesting2d 完全独立: 不导入 nesting2d, 用逐格集合运算与
朴素递归枚举全部整刀切法, 只回答"是否可切"与"(最少刀数, 最短总长)"。
仅用于小规模交叉验证。
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Tuple

Cell = Tuple[int, int]
Rect = Tuple[int, int, int, int]  # (x0, y0, x1, y1) 半开区间


def _spans_line(cells: FrozenSet[Cell], axis: str, k: int, lo: int, hi: int) -> bool:
    """该零件是否在区间 [lo, hi) 内横跨切线(切线穿过零件内部)。"""
    if axis == "v":
        return any(
            (k - 1, y) in cells and (k, y) in cells for y in range(lo, hi)
        )
    return any(
        (x, k - 1) in cells and (x, k) in cells for x in range(lo, hi)
    )


def best_cut(part_cells, rect: Rect) -> Optional[Tuple[int, int]]:
    """返回 (最少刀数, 最短总切长); 不可整刀返回 None。朴素记忆化递归。"""
    memo: Dict[Rect, Optional[Tuple[int, int]]] = {}

    def solve(rect: Rect) -> Optional[Tuple[int, int]]:
        if rect in memo:
            return memo[rect]
        x0, y0, x1, y1 = rect
        present = [
            cells
            for cells in part_cells
            if any(x0 <= x < x1 and y0 <= y < y1 for x, y in cells)
        ]
        if len(present) <= 1:
            memo[rect] = (0, 0)
            return memo[rect]
        best: Optional[Tuple[int, int]] = None
        for axis in ("v", "h"):
            lo, hi = (x0, x1) if axis == "v" else (y0, y1)
            s_lo, s_hi = (y0, y1) if axis == "v" else (x0, x1)
            for k in range(lo + 1, hi):
                if any(_spans_line(c, axis, k, s_lo, s_hi) for c in present):
                    continue
                if axis == "v":
                    r1, r2 = (x0, y0, k, y1), (k, y0, x1, y1)
                    cut_len = y1 - y0
                else:
                    r1, r2 = (x0, y0, x1, k), (x0, k, x1, y1)
                    cut_len = x1 - x0
                a, b = solve(r1), solve(r2)
                if a is None or b is None:
                    continue
                cand = (1 + a[0] + b[0], cut_len + a[1] + b[1])
                if best is None or cand < best:
                    best = cand
        memo[rect] = best
        return best

    return solve(rect)

