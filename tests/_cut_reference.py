"""切割段生成的独立穷举参考(验收用)。

与被测核心 nesting2d 完全独立: 不导入 nesting2d, 使用不同的算法组织 ——
对每条支撑直线收集全部端点与真交点作为切分参数, 再用"区间中点被哪些
零件的边覆盖"决定每段归属, 而被测实现是沿单条边收集切分点后按原子段
合并。两者只在输出契约层面比较。
仅适用于小规模交叉验证。
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd, lcm
from typing import Dict, List, Sequence, Tuple

Pt = Tuple[Fraction, Fraction]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _cross(u, v):
    return u[0] * v[1] - u[1] * v[0]


def _on_segment(p, a, b):
    if _cross(_sub(b, a), _sub(p, a)) != 0:
        return False
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def _line_id(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    s = lcm(dx.denominator, dy.denominator)
    ix, iy = int(dx * s), int(dy * s)
    g = gcd(abs(ix), abs(iy))
    ux, uy = ix // g, iy // g
    if ux < 0 or (ux == 0 and uy < 0):
        ux, uy = -ux, -uy
    c = uy * a[0] - ux * a[1]
    return (ux, uy, c)


def reference_cut_segments(polygons, translations, kerfs):
    """polygons[i]: (外轮廓, 孔...) 环序列; 环为点列表(顺序任意)。

    返回排序后的切割段列表: (p1, p2, parts_tuple, kerf)。
    """
    # edges: dict line_id -> list of (a, b, part, kerf)
    edges: Dict[tuple, List[tuple]] = {}
    for i, (rings, (tx, ty)) in enumerate(zip(polygons, translations)):
        for ring in rings:
            moved = [(Fraction(x) + tx, Fraction(y) + ty) for x, y in ring]
            n = len(moved)
            for k in range(n):
                a, b = moved[k], moved[(k + 1) % n]
                if a > b:
                    a, b = b, a
                edges.setdefault(_line_id(a, b), []).append(
                    (a, b, i, kerfs[i])
                )

    all_edges = [e for lst in edges.values() for e in lst]
    lines = list(edges.keys())

    # 每条直线: 切分参数集合 t=dot(p,u), 并保留 t->实际点
    line_ts: Dict[tuple, Dict[Fraction, Pt]] = {lid: {} for lid in lines}
    for lid, lst in edges.items():
        ux, uy, _ = lid
        for a, b, _i, _k in lst:
            line_ts[lid][a[0] * ux + a[1] * uy] = a
            line_ts[lid][b[0] * ux + b[1] * uy] = b

    # 真交点(非平行边, 交点在两边内部或边界): 加入各自直线的切分点
    for x in range(len(all_edges)):
        a, b, _i, _k = all_edges[x]
        la = _line_id(a, b)
        ua = (b[0] - a[0], b[1] - a[1])
        for y in range(x + 1, len(all_edges)):
            c, d, _j, _k2 = all_edges[y]
            lb = _line_id(c, d)
            if la == lb:
                continue
            ub = (d[0] - c[0], d[1] - c[1])
            den = _cross(ua, ub)
            if den == 0:
                continue
            t = _cross(_sub(c, a), ub) / den
            u = _cross(_sub(c, a), ua) / den
            if 0 <= t <= 1 and 0 <= u <= 1:
                p = (a[0] + t * ua[0], a[1] + t * ua[1])
                if 0 < t < 1:
                    ux, uy, _ = la
                    line_ts[la][p[0] * ux + p[1] * uy] = p
                if 0 < u < 1:
                    ux, uy, _ = lb
                    line_ts[lb][p[0] * ux + p[1] * uy] = p

    # 逐直线在相邻切分点之间用中点判定覆盖
    # raw intervals: (line, t_lo, t_hi, frozenset(parts), kerf)
    raw_intervals = []
    for lid, lst in edges.items():
        ux, uy, _ = lid
        ts = sorted(line_ts[lid])
        for q in range(len(ts) - 1):
            t1, t2 = ts[q], ts[q + 1]
            if t1 == t2:
                continue
            p1, p2 = line_ts[lid][t1], line_ts[lid][t2]
            mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            covers = set()
            for a, b, i, k in lst:
                if _on_segment(mid, a, b):
                    covers.add((i, k))
            if not covers:
                continue
            by_kerf: Dict[Fraction, set] = {}
            for i, k in covers:
                by_kerf.setdefault(k, set()).add(i)
            for k, idxs in by_kerf.items():
                raw_intervals.append(
                    (lid, t1, t2, tuple(sorted(idxs)), k)
                )

    # 合并同一直线相邻且属性相同的区间; 连接点若属于其他直线则断开
    point_lines: Dict[Pt, set] = {}
    for lid, t1, t2, _parts, _k in raw_intervals:
        point_lines.setdefault(line_ts[lid][t1], set()).add(lid)
        point_lines.setdefault(line_ts[lid][t2], set()).add(lid)

    out = []
    by_line: Dict[tuple, list] = {}
    for iv in raw_intervals:
        by_line.setdefault(iv[0], []).append(iv)
    for lid, ivs in by_line.items():
        groups: Dict[Tuple[Tuple[int, ...], Fraction], list] = {}
        for iv in ivs:
            groups.setdefault((iv[3], iv[4]), []).append(iv)
        for (parts, k), g in groups.items():
            g.sort(key=lambda iv: iv[1])
            cur_lo, cur_hi = g[0][1], g[0][2]
            for iv in g[1:]:
                junction = line_ts[lid][iv[1]]
                if iv[1] == cur_hi and len(point_lines[junction]) == 1:
                    cur_hi = iv[2]
                else:
                    out.append(_emit(line_ts[lid], cur_lo, cur_hi, parts, k))
                    cur_lo, cur_hi = iv[1], iv[2]
            out.append(_emit(line_ts[lid], cur_lo, cur_hi, parts, k))
    out.sort(key=lambda z: (z[0], z[1], z[3]))
    return out


def _emit(tpoint, t1, t2, parts, k):
    p1, p2 = tpoint[t1], tpoint[t2]
    if p1 > p2:
        p1, p2 = p2, p1
    return (p1, p2, parts, k)
