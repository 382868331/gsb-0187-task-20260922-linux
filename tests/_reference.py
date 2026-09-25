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


def solve(board, parts, allow_reflection: bool = True, fixed=None):
    """穷举求解。board/parts 为 nesting2d 的 Board/Part(只读其公开字段)。

    fixed: 可选的锁定放置, 形如 {part_index: (x, y, rot, mirrored)} 或
    (part_index, x, y, rot, mirrored) 的序列(独立于被测核心解析, 只接受
    普通 int/bool/tuple, 不依赖 nesting2d.FixedPlacement)。锁定零件只保留
    其唯一指定候选; 锁定候选非法(板内/禁放格/互相冲突)时该高度无候选。

    返回 (height, rotation_count, lex_key, layout) 或 None(不可行)。
    rotation_count 只统计未锁定零件; layout 按零件下标排序, 含锁定件。
    """
    width, max_h = board.width, board.height
    forbidden = set(board.forbidden)
    n = len(parts)

    fixed_map = {}
    if fixed is not None:
        if isinstance(fixed, dict):
            fixed_iter = ((i, v) for i, v in fixed.items())
        else:
            fixed_iter = (
                (v[0], v[1:]) for v in fixed
            )
        for i, v in fixed_iter:
            fixed_map[int(i)] = (int(v[0]), int(v[1]), int(v[2]), bool(v[3]))

    ori_list = [
        orientations(p.cells, p.rotations, allow_reflection and p.allow_reflection)
        for p in parts
    ]

    best = None  # (height, rot_count, lex_key, layout)
    for h in range(1, max_h + 1):
        # 每个零件在该高度下的全部候选: (x, y, rot, mirrored, cells)
        cand_per_part: List[List[Tuple]] = []
        fixed_conflict = False
        for pi, ori in enumerate(ori_list):
            cands = []
            if pi in fixed_map:
                fx, fy, frot, fmir = fixed_map[pi]
                part = parts[pi]
                # 独立判定锁定方向是否被允许(不查去重后的 ori 字典:
                # 镜像可能与某旋转几何同余, 但调用方指定的 rot/mir 元组
                # 本身必须合法 —— 与被测核心的锁定校验口径一致)。
                allowed = (
                    frot in part.rotations
                    and (not fmir or (allow_reflection and part.allow_reflection))
                )
                if not allowed:
                    fixed_conflict = True
                else:
                    shape = transform(part.cells, frot, fmir)
                    sw = max(x for x, _ in shape) + 1
                    sh = max(y for _, y in shape) + 1
                    if 0 <= fx and 0 <= fy and fx + sw <= width and fy + sh <= h:
                        placed = tuple(
                            sorted((cx + fx, cy + fy) for cx, cy in shape)
                        )
                        if not any(c in forbidden for c in placed):
                            cands.append((fx, fy, frot, fmir, placed))
                    # 越界/覆盖禁放格 => 无候选, 该高度不可行
            else:
                for shape, (rot, mir) in ori.items():
                    for x, y, placed in placements_at_height(
                        shape, width, h, forbidden
                    ):
                        cands.append((x, y, rot, mir, placed))
            cand_per_part.append(cands)
        if fixed_conflict or any(not c for c in cand_per_part):
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
                # 第二目标只统计本次可移动(未锁定)零件的旋转
                add_cost = i not in fixed_map and cand[2] != 0
                bt(i + 1, rot_cost + (1 if add_cost else 0))
                chosen[i] = None

        bt(0, 0)
        if best_layout_at_h[0] is not None:
            return (h, best_key_at_h[0][0], best_key_at_h[0][1], best_layout_at_h[0])
    return None
