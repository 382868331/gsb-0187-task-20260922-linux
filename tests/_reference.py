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


def solve(board, parts, allow_reflection: bool = True, locked=()):
    """穷举求解。board/parts 为 nesting2d 的 Board/Part(只读其公开字段)。

    locked: LockedPlacement 序列(假定已被核心验证合法; 参考实现只做
    独立的几何展开, 不重复规格校验)。锁定件位置固定, 只穷举自由件。
    目标: 最小高度(含锁定件最高占用) -> 自由件旋转次数 -> 完整布局字典序。

    返回 (height, rotation_count, lex_key, layout) 或 None(不可行)。
    layout: tuple of (x, y, rot, mirrored), 按零件下标排序。
    """
    width, max_h = board.width, board.height
    forbidden = set(board.forbidden)
    n = len(parts)

    locked_by_index: Dict[int, Tuple] = {}
    locked_cells_list: List[Tuple[Cell, ...]] = []
    locked_top = 0
    for lk in locked:
        i = lk.part_index
        shape = transform(parts[i].cells, lk.rotation_degrees, lk.mirrored)
        cells = tuple(sorted((cx + lk.x, cy + lk.y) for cx, cy in shape))
        locked_by_index[i] = (lk.x, lk.y, lk.rotation_degrees, lk.mirrored, cells)
        locked_cells_list.append(cells)
        locked_top = max(locked_top, max(y for _, y in cells) + 1)
    free_indices = [i for i in range(n) if i not in locked_by_index]
    ori_map = {
        i: orientations(
            parts[i].cells, parts[i].rotations,
            allow_reflection and parts[i].allow_reflection,
        )
        for i in free_indices
    }

    for h in range(max(1, locked_top), max_h + 1):
        # 每个自由零件在该高度下的全部候选: (x, y, rot, mirrored, cells)
        cand_per_part: List[List[Tuple]] = []
        for i in free_indices:
            cands = []
            for shape, (rot, mir) in ori_map[i].items():
                for x, y, placed in placements_at_height(shape, width, h, forbidden):
                    if all(chebyshev_ok(placed, lc) for lc in locked_cells_list):
                        cands.append((x, y, rot, mir, placed))
            cand_per_part.append(cands)
        if any(not c for c in cand_per_part):
            continue
        m = len(free_indices)
        chosen: List[Optional[Tuple]] = [None] * m
        best_key_at_h: List[Optional[Tuple]] = [None]
        best_layout_at_h: List[Optional[Tuple]] = [None]

        def key_of(layout):
            return tuple((x, y, (r, mm)) for x, y, r, mm in layout)

        def bt(k: int, rot_cost: int) -> None:
            if best_key_at_h[0] is not None and rot_cost > best_key_at_h[0][0]:
                return
            if k == m:
                full: List[Optional[Tuple]] = [None] * n
                for idx, entry in locked_by_index.items():
                    full[idx] = entry[:4]
                for idx, c in zip(free_indices, chosen):
                    full[idx] = (c[0], c[1], c[2], c[3])
                layout = tuple(full)
                key = (rot_cost, key_of(layout))
                if best_key_at_h[0] is None or key < best_key_at_h[0]:
                    best_key_at_h[0] = key
                    best_layout_at_h[0] = layout
                return
            for cand in cand_per_part[k]:
                cells = cand[4]
                ok = True
                for j in range(k):
                    if not chebyshev_ok(cells, chosen[j][4]):
                        ok = False
                        break
                if not ok:
                    continue
                chosen[k] = cand
                bt(k + 1, rot_cost + (1 if cand[2] != 0 else 0))
                chosen[k] = None

        bt(0, 0)
        if best_layout_at_h[0] is not None:
            return (h, best_key_at_h[0][0], best_key_at_h[0][1], best_layout_at_h[0])
    return None
