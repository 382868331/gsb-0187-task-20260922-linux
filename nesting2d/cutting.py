"""nesting2d.cutting — 对已有合法排样结果生成切割段(有理坐标直边)。

输入
====
``generate_cut_segments(board, parts, placements, kerf_width=0)`` 接受一组
已经排好的放置(与 ``pack`` 的输出数据结构相同: ``Placement``)。这里的
"合法"指 **几何排样合法**, 不施加排样优化契约里的 Chebyshev 一格边距:

* 每个零件恰好放置一次, 方向被该零件允许;
* 格子全部在板内且不覆盖禁放格;
* 不同零件的格子集合互不相交(但允许 **共用边界边**, 这正是共边切割的
  前提; ``pack`` 带边距的输出是它的严格子集)。

处理范围
========
* **有理坐标直边**: 零件为整数网格多边形, 旋转只取 0/90/180/270 度,
  故全部轮廓边都是轴对齐直边, 端点为整数(有理数的特例);
  切缝补偿 ``kerf_width`` 接受 ``int`` 或 ``fractions.Fraction``(非负),
  全程精确有理运算, 不接受 float。
* **共边只切一次**: 几何完全重合(同一支撑线 + 同一区间)且切缝补偿相同
  的边合并为一条 ``CutSegment``, ``part_ids`` 保留全部相关零件。
* **部分重合不错误去重**: 同一直线上的边在所有端点处切成原子区间再按
  归属合并; 只有真正重合的原子段被合并, 仅端点相接(T 接/拐角)不合并,
  同一零件同一直线上被打断的边也不会跨不同归属错误连通。
* **内外轮廓**: 逐零件用泛洪区分外轮廓与孔洞内轮廓; 共边两侧角色不同
  时(如某零件的外轮廓恰好贴合另一零件的孔边), 仍只切一次,
  ``contours`` 同时保留 ``("inner", "outer")``(``contour`` 属性报告
  ``"shared"``)。
* **切缝补偿不同不合并**: 重合边上两侧零件的 kerf 不同时, 输出两条几何
  相同、kerf 不同的切割段(物理上是两次不同补偿的切割)。

端点顺序固定: 每条段的端点按 ``(x, y)`` 字典序排列(小端在前);
``CutPlan.segments`` 顺序确定(水平在前、竖直在后, 再按支撑线坐标与
区间排序), 可直接做字典序比较。

非法输入抛 ``TypeError``(类型问题, 含 bool 充整数/float)或
``InvalidSpecError``(取值域/几何冲突); 函数无副作用, 不修改任何输入,
失败时不会产生部分结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Sequence, Set, Tuple

from . import (
    Board,
    InvalidSpecError,
    Part,
    Placement,
    _require_int,
    oriented_cells,
)

__all__ = [
    "CutSegment",
    "CutPlan",
    "generate_cut_segments",
]

# (kind, c, lo, hi): kind=0 水平(支撑线 y=c, 区间为 x 范围);
# kind=1 竖直(支撑线 x=c, 区间为 y 范围)。
_Run = Tuple[int, int, int, int, str]
Cell = Tuple[int, int]


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CutSegment:
    """一条直边切割段。

    ``(x1, y1)``–``(x2, y2)`` 为端点, 固定排序保证
    ``(x1, y1) <= (x2, y2)``(端点处的 T 接也算端点)。
    ``kerf_width``: 本条切割使用的切缝补偿(非负有理数)。
    ``part_ids``: 本条段涉及的全部零件下标(共边时多于一个)。
    ``contours``: 涉及的轮廓角色, 取值子集
    ``("outer", "inner")`` 的排序元组; 两者同时出现表示该共边一侧是
    外轮廓、另一侧是孔轮廓。
    """

    x1: int
    y1: int
    x2: int
    y2: int
    kerf_width: Fraction
    part_ids: frozenset
    contours: Tuple[str, ...]

    @property
    def length(self) -> Fraction:
        """段长(轴对齐, 等于两端点的坐标差)。"""
        return Fraction(self.x2 - self.x1 + (self.y2 - self.y1))

    @property
    def contour(self) -> str:
        """聚合角色: ``"outer"`` / ``"inner"`` / ``"shared"``(内外共有)。"""
        if "outer" in self.contours and "inner" in self.contours:
            return "shared"
        return self.contours[0]


@dataclass(frozen=True)
class CutPlan:
    """切割方案。

    ``segments``: 去重/拆分后的全部切割段(顺序确定)。
    ``total_length``: 实际切割总长度 —— 每条输出段计一次, 因此同一条几何
    边因切缝补偿不同而切两次时计两次长度。
    """

    segments: Tuple[CutSegment, ...]
    total_length: Fraction

    def __len__(self) -> int:
        return len(self.segments)


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def _require_rational_nonneg(value: Any, path: str) -> Fraction:
    """要求非负有理数(int 或 Fraction)。拒绝 bool/float/其他类型。"""
    if isinstance(value, bool):
        raise TypeError(f"{path}: 需要 int/Fraction, 不允许 bool (得到 {value!r})")
    if isinstance(value, int):
        frac = Fraction(value)
    elif isinstance(value, Fraction):
        frac = value
    elif isinstance(value, float):
        raise TypeError(
            f"{path}: 需要 int/Fraction, 不接受 float {value!r}"
            "(切缝补偿必须精确有理; 请用 Fraction)"
        )
    else:
        raise TypeError(
            f"{path}: 需要 int/Fraction, 得到 {type(value).__name__}"
        )
    if frac < 0:
        raise InvalidSpecError(f"{path}: 切缝补偿不能为负 (得到 {value!r})")
    return frac


def _validate_cut_input(
    board: Board,
    parts: Sequence[Part],
    placements: Sequence[Placement],
    kerf_width: Any,
) -> Tuple[Tuple[frozenset, ...], Tuple[Fraction, ...]]:
    """校验全部输入, 返回 (每件放置后的绝对格子集, 每件 kerf)。

    任何非法输入直接抛异常; 不修改调用方对象。
    """
    if not isinstance(board, Board):
        raise TypeError(f"board: 需要 Board, 得到 {type(board).__name__}")
    if isinstance(parts, (str, bytes)):
        raise TypeError("parts: 需要 Part 序列")
    try:
        parts = tuple(parts)
    except TypeError:
        raise TypeError("parts: 需要 Part 序列") from None
    for i, p in enumerate(parts):
        if not isinstance(p, Part):
            raise TypeError(f"parts[{i}]: 需要 Part, 得到 {type(p).__name__}")
    if not (1 <= len(parts) <= 12):
        raise InvalidSpecError(
            f"parts: 零件数 {len(parts)} 超出范围 [1, 12]"
        )

    if isinstance(placements, (str, bytes)):
        raise TypeError("placements: 需要 Placement 序列")
    try:
        placements = tuple(placements)
    except TypeError:
        raise TypeError("placements: 需要 Placement 序列") from None
    for i, pl in enumerate(placements):
        if not isinstance(pl, Placement):
            raise TypeError(
                f"placements[{i}]: 需要 Placement, 得到 {type(pl).__name__}"
            )
    n = len(parts)
    if len(placements) != n:
        raise InvalidSpecError(
            f"placements: 放置数 {len(placements)} 与零件数 {n} 不一致"
            "(每个零件必须恰好放置一次)"
        )

    seen_idx: Set[int] = set()
    placed: List[frozenset] = [frozenset()] * n
    for i, pl in enumerate(placements):
        idx = _require_int(pl.part_index, f"placements[{i}].part_index")
        if not (0 <= idx < n):
            raise InvalidSpecError(
                f"placements[{i}].part_index: {idx} 超出零件下标范围 [0, {n - 1}]"
            )
        if idx in seen_idx:
            raise InvalidSpecError(f"placements: 零件 {idx} 被重复放置")
        seen_idx.add(idx)
        part = parts[idx]
        rot = _require_int(pl.rotation_degrees, f"placements[{i}].rotation_degrees")
        if rot not in (0, 90, 180, 270):
            raise InvalidSpecError(
                f"placements[{i}].rotation_degrees: {rot} 不是 0/90/180/270"
            )
        if rot not in part.rotations:
            raise InvalidSpecError(
                f"零件 {idx}: 旋转角 {rot} 不在允许集合 {sorted(part.rotations)} 内"
            )
        if not isinstance(pl.mirrored, bool):
            raise TypeError(
                f"placements[{i}].mirrored: 需要 bool, 得到 "
                f"{type(pl.mirrored).__name__}"
            )
        if pl.mirrored and not part.allow_reflection:
            raise InvalidSpecError(f"零件 {idx}: 使用了镜像但该零件禁用镜像")
        px = _require_int(pl.x, f"placements[{i}].x")
        py = _require_int(pl.y, f"placements[{i}].y")
        shape = oriented_cells(part.cells, rot, pl.mirrored)
        cells = frozenset((cx + px, cy + py) for cx, cy in shape)
        for cx, cy in cells:
            if not (0 <= cx < board.width and 0 <= cy < board.height):
                raise InvalidSpecError(
                    f"零件 {idx}: 放置后格子 ({cx}, {cy}) 超出板材范围 "
                    f"{board.width}x{board.height}"
                )
            if (cx, cy) in board.forbidden:
                raise InvalidSpecError(
                    f"零件 {idx}: 放置后格子 ({cx}, {cy}) 覆盖禁放格"
                )
        placed[idx] = cells

    # 零件之间格子不得重叠(共用边界允许)
    owner: Dict[Cell, int] = {}
    for idx, cells in enumerate(placed):
        for c in cells:
            other = owner.get(c)
            if other is not None:
                raise InvalidSpecError(
                    f"零件 {idx} 与零件 {other} 在格子 {c} 重叠"
                )
            owner[c] = idx

    # 切缝补偿: 标量(全体相同)或按零件下标的序列
    if isinstance(kerf_width, (str, bytes)):
        raise TypeError("kerf_width: 需要非负 int/Fraction 或其按零件的序列")
    if isinstance(kerf_width, (int, Fraction)) and not isinstance(kerf_width, bool):
        k = _require_rational_nonneg(kerf_width, "kerf_width")
        kerfs = tuple(k for _ in range(n))
    else:
        try:
            seq = tuple(kerf_width)
        except TypeError:
            raise TypeError(
                "kerf_width: 需要非负 int/Fraction 或长度等于零件数的序列"
            ) from None
        if len(seq) != n:
            raise InvalidSpecError(
                f"kerf_width: 序列长度 {len(seq)} 与零件数 {n} 不一致"
            )
        kerfs = tuple(
            _require_rational_nonneg(v, f"kerf_width[{i}]") for i, v in enumerate(seq)
        )

    return tuple(placed), kerfs


# ---------------------------------------------------------------------------
# 轮廓提取(内外区分 + 直边合并)
# ---------------------------------------------------------------------------


def _exterior_components(cells: Set[Cell]) -> Dict[Cell, int]:
    """把每个零件外的邻域格子标记连通分量: 0 = 外部, >=1 = 孔洞。

    在 cells 包围盒外扩一格的矩形内做 4-邻接泛洪; 从矩形角点可达的空格
    即外部, 其余空格按连通分量编号为各个孔洞。
    """
    min_x = min(x for x, _ in cells) - 1
    max_x = max(x for x, _ in cells) + 1
    min_y = min(y for _, y in cells) - 1
    max_y = max(y for _, y in cells) + 1

    def inside(c: Cell) -> bool:
        x, y = c
        return min_x <= x <= max_x and min_y <= y <= max_y

    comp: Dict[Cell, int] = {}
    # 外部: 从外扩矩形四角一起泛洪(矩形边界整圈连通, 角点足够)
    stack = [(min_x, min_y), (max_x, min_y), (min_x, max_y), (max_x, max_y)]
    while stack:
        c = stack.pop()
        if c in comp or not inside(c) or c in cells:
            continue
        comp[c] = 0
        x, y = c
        stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
    # 剩余空格 = 孔洞, 逐分量编号
    hole_id = 0
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            c = (x, y)
            if c in cells or c in comp:
                continue
            hole_id += 1
            stack = [c]
            while stack:
                q = stack.pop()
                if q in comp or not inside(q) or q in cells:
                    continue
                comp[q] = hole_id
                qx, qy = q
                stack.extend(
                    [(qx + 1, qy), (qx - 1, qy), (qx, qy + 1), (qx, qy - 1)]
                )
    return comp


def _contour_runs(cells: Set[Cell]) -> List[_Run]:
    """提取一个零件的全部极大直边轮廓段(未跨零件合并)。

    返回 (kind, c, lo, hi, role): kind 0 水平 / 1 竖直;
    role "outer"/"inner"。同一条支撑线上同一轮廓分量的相邻单位边
    合并为极大段(仅端点相接的不同分量不合并)。
    """
    comp = _exterior_components(cells)
    # (kind, c, component) -> 单位区间列表
    intervals: Dict[Tuple[int, int, int], List[Tuple[int, int, str]]] = {}
    for x, y in cells:
        neighbors = (
            (x + 1, y, 1, x + 1, y),       # 东: 竖直 x=x+1, y..y+1
            (x, y + 1, 0, y + 1, x),       # 北: 水平 y=y+1, x..x+1
            (x - 1, y, 1, x, y),           # 西: 竖直 x=x, y..y+1
            (x, y - 1, 0, y, x),           # 南: 水平 y=y, x..x+1
        )
        for nx, ny, kind, c, lo in neighbors:
            if (nx, ny) in cells:
                continue
            cid = comp[(nx, ny)]
            role = "outer" if cid == 0 else "inner"
            intervals.setdefault((kind, c, cid), []).append((lo, lo + 1, role))

    runs: List[_Run] = []
    for (kind, c, _cid), units in intervals.items():
        units.sort()
        cur_lo, cur_hi, role = units[0]
        for lo, hi, _ in units[1:]:
            if lo == cur_hi:  # 同一轮廓分量上首尾相接 -> 合并
                cur_hi = hi
            else:
                runs.append((kind, c, cur_lo, cur_hi, role))
                cur_lo, cur_hi = lo, hi
        runs.append((kind, c, cur_lo, cur_hi, role))
    return runs


# ---------------------------------------------------------------------------
# 原子区间叠加(共边/部分重合/切缝分组)
# ---------------------------------------------------------------------------


def _overlay(
    placed: Sequence[frozenset], kerfs: Sequence[Fraction]
) -> List[CutSegment]:
    """把各零件的极大直边叠加为去重后的切割段。

    每条支撑线上收集全部端点作为分割点; 每个原子开区间按 kerf 分组
    (同 kerf 的所有零件合为一条), 再把相邻且归属完全相同的原子区间
    合并回极大段。
    """
    # (kind, c) -> list of (lo, hi, kerf, part, role)
    lines: Dict[Tuple[int, int], List[Tuple[int, int, Fraction, int, str]]] = {}
    for part_id, cells in enumerate(placed):
        for kind, c, lo, hi, role in _contour_runs(set(cells)):
            lines.setdefault((kind, c), []).append(
                (lo, hi, kerfs[part_id], part_id, role)
            )

    segments: List[CutSegment] = []
    for (kind, c), recs in lines.items():
        breaks_set: Set[int] = set()
        for lo, hi, _k, _p, _r in recs:
            breaks_set.add(lo)
            breaks_set.add(hi)
        breaks = sorted(breaks_set)
        # 每个原子区间的归属标签: kerf -> (parts, roles)
        atoms: List[Tuple[int, int, Dict[Fraction, Tuple[Set[int], Set[str]]]]] = []
        for bi in range(len(breaks) - 1):
            lo, hi = breaks[bi], breaks[bi + 1]
            groups: Dict[Fraction, Tuple[Set[int], Set[str]]] = {}
            for r_lo, r_hi, r_k, r_p, r_r in recs:
                if r_lo <= lo and r_hi >= hi:
                    entry = groups.get(r_k)
                    if entry is None:
                        entry = (set(), set())
                        groups[r_k] = entry
                    entry[0].add(r_p)
                    entry[1].add(r_r)
            atoms.append((lo, hi, groups))

        # 相邻原子若"每个 kerf 组的 (零件集, 角色集)"完全相同则合并
        def label(groups):
            return tuple(
                (k, tuple(sorted(ps)), tuple(sorted(rs)))
                for k, (ps, rs) in sorted(groups.items())
            )

        run_lo = breaks[0]
        run_hi = breaks[1]
        run_label = label(atoms[0][2])
        run_groups = atoms[0][2]
        for bi in range(1, len(atoms)):
            lo, hi, groups = atoms[bi]
            lab = label(groups)
            if lab == run_label:
                run_hi = hi
                continue
            _emit(segments, kind, c, run_lo, run_hi, run_groups)
            run_lo, run_hi, run_label, run_groups = lo, hi, lab, groups
        _emit(segments, kind, c, run_lo, run_hi, run_groups)

    return segments


def _emit(
    out: List[CutSegment],
    kind: int,
    c: int,
    lo: int,
    hi: int,
    groups: Dict[Fraction, Tuple[Set[int], Set[str]]],
) -> None:
    for kerf, (parts, roles) in groups.items():
        if kind == 0:  # 水平: 支撑线 y=c
            p1, p2 = (lo, c), (hi, c)
        else:          # 竖直: 支撑线 x=c
            p1, p2 = (c, lo), (c, hi)
        (x1, y1), (x2, y2) = sorted((p1, p2))
        out.append(
            CutSegment(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                kerf_width=kerf,
                part_ids=frozenset(parts),
                contours=tuple(sorted(roles)),
            )
        )


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------


def generate_cut_segments(
    board: Board,
    parts: Sequence[Part],
    placements: Sequence[Placement],
    kerf_width: Any = 0,
) -> CutPlan:
    """对已有合法排样结果生成切割段。

    参数
    ----
    board: ``Board`` 实例(提供板边界与禁放格)。
    parts: ``Part`` 序列, 长度 1..12。
    placements: ``Placement`` 序列, 每个零件恰好一次; 放置方向必须被
        对应零件允许, 格子在板内、不覆盖禁放格、零件间不重叠
        (允许共用边界)。
    kerf_width: 切缝补偿。全体统一时给非负 ``int``/``Fraction``(默认 0);
        需要按零件区分时给长度等于零件数的序列。完全重合的边只在
        kerf 相同时合并。

    返回 ``CutPlan``; 输入非法时抛 ``TypeError`` / ``InvalidSpecError``,
    且不修改任何输入、不产生部分结果。
    """
    placed, kerfs = _validate_cut_input(board, parts, placements, kerf_width)

    # 重新做一次确定性的排序键(简单清晰): 水平(y, x0, x1), 竖直(x, y0, y1)
    segments = _overlay(placed, kerfs)
    segments.sort(
        key=lambda s: (
            0 if s.y1 == s.y2 else 1,
            s.y1 if s.y1 == s.y2 else s.x1,
            s.x1 if s.y1 == s.y2 else s.y1,
            s.x2 if s.y1 == s.y2 else s.y2,
            s.kerf_width,
        )
    )
    total = Fraction(0)
    for seg in segments:
        total += seg.length
    return CutPlan(segments=tuple(segments), total_length=total)
