"""独立穷举参考实现(验收用)。

与被测核心 nesting2d 完全独立: 不导入 nesting2d, 不使用位集,
用独立的坐标变换与逐格 Chebyshev 距离检查, 在小板上穷举全部放置。
仅用于测试交叉验证(接口边界 20x20/12 件对穷举过大, 这里只跑小规模)。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

Cell = Tuple[int, int]


def normalize(cells) -> Tuple[Cell, ...]:
    cells = list(cells)
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    return tuple(sorted((x - min_x, y - min_y) for x, y in cells))


def transform(cells, rot: int, mirrored: bool) -> Tuple[Cell, ...]:
    """独立的旋转/镜像实现: 先镜像 (x,y)->(-x,y), 再逆时针旋转 rot 度。"""
    out = []
    for x, y in cells:
        if mirrored:
            x = -x
        for _ in range(rot // 90):
            x, y = -y, x
        out.append((x, y))
    return normalize(out)


def orientations(cells, rotations, allow_reflection: bool) -> Dict[Tuple[Cell, ...], Tuple[int, bool]]:
    """{归一化形状: (rot, mirrored)} 去重; 与核心实现无关的独立写法。"""
    seen: Dict[Tuple[Cell, ...], Tuple[int, bool]] = {}
    for mirrored in ([False, True] if allow_reflection else [False]):
        for rot in (0, 90, 180, 270):
            if rot not in rotations:
                continue
            shape = transform(cells, rot, mirrored)
            if shape not in seen:
                seen[shape] = (rot, mirrored)
    return seen


def chebyshev_ok(cells_a, cells_b) -> bool:
    """两组格子两两 Chebyshev 距离 >= 2(即一格边距)。"""
    for ax, ay in cells_a:
        for bx, by in cells_b:
            if max(abs(ax - bx), abs(ay - by)) < 2:
                return False
    return True


def placements_at_height(shape, width, height, forbidden):
    """一个归一化形状在 width x height 区域内的全部放置(绝对格子元组)。"""
    sw = max(x for x, _ in shape) + 1
    sh = max(y for _, y in shape) + 1
    out = []
    for y in range(height - sh + 1):
        for x in range(width - sw + 1):
            placed = tuple(sorted((cx + x, cy + y) for cx, cy in shape))
            if any(c in forbidden for c in placed):
                continue
            out.append((x, y, placed))
    return out


def solve(board, parts, allow_reflection: bool = True):
    """穷举求解。board/parts 为 nesting2d 的 Board/Part(只读其公开字段)。

    返回 (height, rotation_count, lex_key, layout) 或 None(不可行)。
    layout: tuple of (x, y, rot, mirrored), 按零件下标排序。
    """
    width, max_h = board.width, board.height
    forbidden = set(board.forbidden)
    n = len(parts)
    ori_list = [
        orientations(p.cells, p.rotations, allow_reflection and p.allow_reflection)
        for p in parts
    ]

    best = None  # (height, rot_count, lex_key, layout)
    for h in range(1, max_h + 1):
        # 每个零件在该高度下的全部候选: (x, y, rot, mirrored, cells)
        cand_per_part: List[List[Tuple]] = []
        for ori in ori_list:
            cands = []
            for shape, (rot, mir) in ori.items():
                for x, y, placed in placements_at_height(shape, width, h, forbidden):
                    cands.append((x, y, rot, mir, placed))
            cand_per_part.append(cands)
        if any(not c for c in cand_per_part):
            continue
        chosen: List[Optional[Tuple]] = [None] * n
        best_key_at_h: List[Optional[Tuple]] = [None]
        best_layout_at_h: List[Optional[Tuple]] = [None]

        def key_of(layout):
            return tuple((x, y, (r, m)) for x, y, r, m in layout)

        def bt(i: int, rot_cost: int) -> None:
            if best_key_at_h[0] is not None and rot_cost > best_key_at_h[0][0]:
                return
            if i == n:
                layout = tuple((c[0], c[1], c[2], c[3]) for c in chosen)
                key = (rot_cost, key_of(layout))
                if best_key_at_h[0] is None or key < best_key_at_h[0]:
                    best_key_at_h[0] = key
                    best_layout_at_h[0] = layout
                return
            for cand in cand_per_part[i]:
                cells = cand[4]
                ok = True
                for j in range(i):
                    if not chebyshev_ok(cells, chosen[j][4]):
                        ok = False
                        break
                if not ok:
                    continue
                chosen[i] = cand
                bt(i + 1, rot_cost + (1 if cand[2] != 0 else 0))
                chosen[i] = None

        bt(0, 0)
        if best_layout_at_h[0] is not None:
            return (h, best_key_at_h[0][0], best_key_at_h[0][1], best_layout_at_h[0])
    return None


def guillotine_min_cuts(bboxes, width, height):
    """独立的整刀可切性参考: 纯递归枚举全部可用切线, 不用记忆化。

    bboxes: 每零件的绝对包围盒 (x0, y0, x1, y1)(半开区间)。
    返回 (最少刀数, 最短总切长); 不可切返回 None。
    只适用于小规模交叉验证(指数级, 无缓存)。
    """

    def solve(rect):
        x0, y0, x1, y1 = rect
        inside = [
            b for b in bboxes
            if x0 <= b[0] and b[1] >= y0 and b[2] <= x1 and b[3] <= y1
        ]
        if len(inside) <= 1:
            return (0, 0)
        best = None
        for c in range(x0 + 1, x1):
            if any(b[0] < c < b[2] for b in inside):
                continue
            left = solve((x0, y0, c, y1))
            right = solve((c, y0, x1, y1))
            if left is None or right is None:
                continue
            cand = (1 + left[0] + right[0],
                    (y1 - y0) + left[1] + right[1])
            if best is None or cand < best:
                best = cand
        for c in range(y0 + 1, y1):
            if any(b[1] < c < b[3] for b in inside):
                continue
            bottom = solve((x0, y0, x1, c))
            top = solve((x0, c, x1, y1))
            if bottom is None or top is None:
                continue
            cand = (1 + bottom[0] + top[0],
                    (x1 - x0) + bottom[1] + top[1])
            if best is None or cand < best:
                best = cand
        return best

    return solve((0, 0, width, height))
