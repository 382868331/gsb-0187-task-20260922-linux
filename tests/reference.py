"""独立穷举参考实现：仅用于测试交叉验证。

约束：不得 import 被测核心模块 nesting；所有几何逻辑（变换、
Chebyshev 边距、字典序比较）在此独立重写。只适用于小板/少零件，
直接枚举所有可行布局后按目标函数取最优，不做任何剪枝优化。
"""

from itertools import product

ROTATIONS = (0, 90, 180, 270)


def normalize(cells):
    min_x = min(c[0] for c in cells)
    min_y = min(c[1] for c in cells)
    return tuple(sorted((c[0] - min_x, c[1] - min_y) for c in cells))


def transform(cells, rot, mirror):
    out = []
    for x, y in cells:
        if mirror:
            x = -x
        if rot == 90:
            x, y = -y, x
        elif rot == 180:
            x, y = -x, -y
        elif rot == 270:
            x, y = y, -x
        out.append((x, y))
    return normalize(out)


def variants(cells, allowed_rotations, allow_mirror):
    """{(归一化形状): 最小 code}，code = 旋转序号 * 2 + 镜像位。"""
    seen = {}
    for rot in allowed_rotations:
        for mir in ((False, True) if allow_mirror else (False,)):
            shape = transform(cells, rot, mir)
            code = ROTATIONS.index(rot) * 2 + (1 if mir else 0)
            if shape not in seen or code < seen[shape]:
                seen[shape] = code
    return seen


def dilate(cells):
    """Chebyshev 一格膨胀（返回格子集合，不做板内裁剪）。"""
    out = set()
    for x, y in cells:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                out.add((x + dx, y + dy))
    return out


def reference_solve(width, height, forbidden, pieces):
    """穷举所有可行布局，返回与核心模块同构的最优结果。

    pieces: [(cells, allowed_rotations, allow_mirror), ...]
    返回 dict(status, height, rotation_count, lex_key, layout)，
    layout 为 [(x, y, code), ...]（按零件输入顺序）。
    """
    forbidden = set(forbidden)
    n = len(pieces)
    var_list = [variants(c, r, m) for c, r, m in pieces]

    # 每个零件的全部合法放置：(x, y, code, frozenset(占用格), frozenset(膨胀区))
    per_piece = []
    for seen in var_list:
        opts = []
        for shape, code in seen.items():
            vw = max(c[0] for c in shape) + 1
            vh = max(c[1] for c in shape) + 1
            for dy in range(height - vh + 1):
                for dx in range(width - vw + 1):
                    occ = frozenset((x + dx, y + dy) for x, y in shape)
                    if occ & forbidden:
                        continue
                    opts.append((dx, dy, code, occ, dilate(occ)))
        per_piece.append(opts)
    if any(not opts for opts in per_piece):
        return {"status": "infeasible", "height": None,
                "rotation_count": None, "lex_key": None, "layout": None}

    best = None  # (height, rot_count, lex_key, layout)

    def consider(layout):
        nonlocal best
        occ_rows = [y for (_, y, _, occ, _) in layout for (x, y) in occ]
        h_used = max(occ_rows) + 1
        rot_count = sum(1 for (_, _, code, _, _) in layout if code >= 2)
        lex_key = tuple((x, y, code) for (x, y, code, _, _) in layout)
        key = (h_used, rot_count, lex_key)
        if best is None or key < best[:3]:
            best = (h_used, rot_count, lex_key,
                    [(x, y, code) for (x, y, code, _, _) in layout])

    def dfs(i, occupied, clearance, layout):
        if i == n:
            consider(layout)
            return
        for (x, y, code, occ, dil) in per_piece[i]:
            if occ & clearance:  # 与已放零件的膨胀区（含自身）相交
                continue
            if dil & occupied:  # 膨胀区覆盖已占格
                continue
            layout.append((x, y, code, occ, dil))
            dfs(i + 1, occupied | occ, clearance | dil, layout)
            layout.pop()

    dfs(0, frozenset(), frozenset(), [])
    if best is None:
        return {"status": "infeasible", "height": None,
                "rotation_count": None, "lex_key": None, "layout": None}
    return {"status": "optimal", "height": best[0],
            "rotation_count": best[1], "lex_key": best[2],
            "layout": best[3]}
