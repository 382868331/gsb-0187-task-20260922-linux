"""nesting2d — 约束二维材料片的无重叠排样(整数网格, 纯标准库)。

问题定义
========
在一块 ``width x height`` 的整数网格板材上放置若干零件(每个零件是一个
4-连通的格子集合), 要求:

* 零件之间互不重叠, 且任意两个不同零件的格子之间的 Chebyshev 距离 >= 2
  (即每个零件按 Chebyshev 一格膨胀后互不相交 —— 零件间保留一格边距);
* 板材边界外 **不** 计额外空间: 零件格子可以紧贴板材边缘;
* 板材上的禁放格(forbidden cells)不得被任何零件格子覆盖
  (边距膨胀只约束零件之间, 不约束零件与禁放格);
* 每个零件可独立指定允许的旋转角度(0/90/180/270 度的子集)以及是否允许
  镜像; 全局 ``allow_reflection`` 为 False 时所有零件禁用镜像。

目标(按优先级):

1. 最小化使用高度 ``h`` (1 <= h <= height, 只使用板材底部 h 行);
2. 在最小高度下最小化旋转次数 ``rotation_count``
   (放置方向旋转角 != 0 的零件个数; 镜像不算旋转);
3. 再按布局坐标字典序最小化: 按零件下标顺序比较每个零件的
   ``(x, y, orientation_code)``, 其中 orientation_code 为
   ``(rotation_degrees, mirrored)``。

坐标系: 格子坐标 ``(x, y)``, ``0 <= x < width``, ``0 <= y < height``,
y 向上增长; 使用底部 h 行即 ``0 <= y < h``。

求解方法: 位集(bitset)回溯搜索。每个候选放置用两个 Python int 位掩码表示
(实际占用掩码 + Chebyshev 一格膨胀掩码), 冲突检测为 O(1) 位运算。
相同零件(形状与允许方向完全一致)在搜索中做对称破缺去重(同组零件的候选
放置下标单调不减); 同一零件由对称性产生的相同方向也去重。搜索预算
``budget`` 以 DFS 节点数计, 全程共享; 预算耗尽时报告当前最优已知解与
已证明的下界, 绝不把未证明的结果标记为最优。

本模块只使用整数运算, 不产生任何浮点结果。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "MAX_BOARD_WIDTH",
    "MAX_BOARD_HEIGHT",
    "MAX_PARTS",
    "MAX_PART_CELLS",
    "MAX_CONTOUR_VERTICES",
    "MAX_POLY_HOLES",
    "DEFAULT_BUDGET",
    "InvalidSpecError",
    "Status",
    "Part",
    "Board",
    "Placement",
    "PackResult",
    "pack",
    "verify_layout",
    "render_layout",
    "oriented_cells",
    "PolyPart",
    "PolyPlacement",
    "CutSegment",
    "CutPlan",
    "cut_segments",
    "cut_segments_for_layout",
]

# ---------------------------------------------------------------------------
# 公开常量(接口边界)
# ---------------------------------------------------------------------------

MAX_BOARD_WIDTH = 20
MAX_BOARD_HEIGHT = 20
MAX_PARTS = 12
MAX_PART_CELLS = 12
DEFAULT_BUDGET = 1_000_000

# 切割段生成的接口边界(多边形零件, 有理坐标直边)
MAX_CONTOUR_VERTICES = 32
MAX_POLY_HOLES = 4

_VALID_ROTATIONS = frozenset({0, 90, 180, 270})


# ---------------------------------------------------------------------------
# 错误语义
# ---------------------------------------------------------------------------


class InvalidSpecError(ValueError):
    """输入规格非法(取值域违规)。

    类型层面的问题(非整数、bool 充当整数、NaN/Infinity 等)抛出内置
    ``TypeError``; 类型正确但违反本题取值域(越界、不连通、重复格等)
    抛出本异常。所有错误消息都包含可定位的字段路径。
    """


def _require_int(value, path: str) -> int:
    """要求 Python int。拒绝 bool; 拒绝 NaN/Infinity 及其他非 int 数值。"""
    if isinstance(value, bool):
        raise TypeError(f"{path}: 需要 int, 不允许 bool (得到 {value!r})")
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise TypeError(f"{path}: 拒绝 NaN/Infinity (得到 {value!r})")
        raise TypeError(f"{path}: 需要 int, 得到 float {value!r}")
    if not isinstance(value, int):
        raise TypeError(f"{path}: 需要 int, 得到 {type(value).__name__}")
    return value


def _parse_cell(cell, path: str) -> Tuple[int, int]:
    """解析一个格子坐标 (x, y)。仅接受二元 tuple/list。"""
    if not isinstance(cell, (tuple, list)) or len(cell) != 2:
        raise TypeError(f"{path}: 需要长度为 2 的 (x, y) 坐标, 得到 {cell!r}")
    x = _require_int(cell[0], f"{path}[0]")
    y = _require_int(cell[1], f"{path}[1]")
    return (x, y)


def _parse_point(point, path: str) -> Tuple[Fraction, Fraction]:
    """解析一个有理坐标点 (x, y); 只接受 int/Fraction, 拒绝 float/bool。"""
    if not isinstance(point, (tuple, list)) or len(point) != 2:
        raise TypeError(
            f"{path}: 需要长度为 2 的 (x, y) 有理坐标, 得到 {point!r}"
        )

    def one(v, sub: str) -> Fraction:
        if isinstance(v, bool):
            raise TypeError(f"{sub}: 不允许 bool 作为坐标 (得到 {v!r})")
        if isinstance(v, int):
            return Fraction(v)
        if isinstance(v, Fraction):
            return Fraction(v)
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                raise TypeError(f"{sub}: 拒绝 NaN/Infinity (得到 {v!r})")
            raise TypeError(
                f"{sub}: 只接受精确有理坐标(int/Fraction), 拒绝 float {v!r}"
            )
        raise TypeError(f"{sub}: 需要 int/Fraction, 得到 {type(v).__name__}")

    return (one(point[0], f"{path}[0]"), one(point[1], f"{path}[1]"))


def _parse_rational(value, path: str) -> Fraction:
    """解析非负有理标量(切缝宽度)。只接受 int/Fraction, 拒绝 float/bool。"""
    if isinstance(value, bool):
        raise TypeError(f"{path}: 不允许 bool (得到 {value!r})")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, Fraction):
        return Fraction(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise TypeError(f"{path}: 拒绝 NaN/Infinity (得到 {value!r})")
        raise TypeError(f"{path}: 只接受精确有理数(int/Fraction), 拒绝 float")
    raise TypeError(f"{path}: 需要 int/Fraction, 得到 {type(value).__name__}")


def _resolve_kerfs(
    n: int, kerf: Any, kerfs: Optional[Dict[int, Any]]
) -> List[Fraction]:
    """解析默认切缝宽度与按零件下标覆盖值, 返回长度 n 的列表。"""
    default_kerf = _parse_rational(kerf, "kerf")
    if default_kerf < 0:
        raise InvalidSpecError(f"kerf: 不能为负, 得到 {default_kerf}")
    kerf_map: Dict[int, Fraction] = {}
    if kerfs is not None:
        if not isinstance(kerfs, dict):
            raise TypeError("kerfs: 需要 dict {part_index: 切缝宽度}")
        for key, val in kerfs.items():
            if not isinstance(key, int) or isinstance(key, bool):
                raise TypeError(f"kerfs 的键需要 int, 得到 {key!r}")
            if not (0 <= key < n):
                raise InvalidSpecError(
                    f"kerfs: 零件下标 {key} 超出范围 [0, {n - 1}]"
                )
            v = _parse_rational(val, f"kerfs[{key}]")
            if v < 0:
                raise InvalidSpecError(f"kerfs[{key}]: 不能为负, 得到 {v}")
            kerf_map[key] = v
    return [kerf_map.get(i, default_kerf) for i in range(n)]


def _normalize_cells(cells: Iterable[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    """平移归一化: min x / min y 移到 0。cells 须为非空可迭代。"""
    cells = list(cells)
    min_x = min(c[0] for c in cells)
    min_y = min(c[1] for c in cells)
    return tuple(sorted((x - min_x, y - min_y) for x, y in cells))


def _is_connected(cells: Iterable[Tuple[int, int]]) -> bool:
    """4-邻接(边相邻)连通性检查。"""
    cells = set(cells)
    if not cells:
        return False
    seen = {next(iter(cells))}
    stack = list(seen)
    while stack:
        x, y = stack.pop()
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen == cells


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Part:
    """一个零件: 连通的格子集合 + 允许的方向。

    ``cells``: 格子坐标可迭代, 每个为 ``(x, y)``; 必须非空、无重复、
    4-连通、格子数 <= MAX_PART_CELLS。构造时归一化为 ``frozenset``;
    零件的绝对坐标无意义(求解时按形状归一化)。

    ``rotations``: 允许的旋转角度子集, 元素取自 {0, 90, 180, 270},
    必须非空; 默认只允许 0 度。

    ``allow_reflection``: 是否允许镜像(须为 bool)。全局
    ``pack(..., allow_reflection=False)`` 会覆盖禁用。
    """

    cells: Iterable[Tuple[int, int]]
    rotations: Iterable[int] = (0,)
    allow_reflection: bool = True

    def __post_init__(self) -> None:
        raw = list(self.cells)
        if not raw:
            raise InvalidSpecError("Part.cells: 至少需要 1 个格子")
        parsed = [_parse_cell(c, f"Part.cells[{i}]") for i, c in enumerate(raw)]
        cell_set = frozenset(parsed)
        if len(cell_set) != len(parsed):
            raise InvalidSpecError("Part.cells: 存在重复格子")
        if len(cell_set) > MAX_PART_CELLS:
            raise InvalidSpecError(
                f"Part.cells: 格子数 {len(cell_set)} 超过上限 {MAX_PART_CELLS}"
            )
        if not _is_connected(cell_set):
            raise InvalidSpecError("Part.cells: 格子集合不连通(需要 4-连通)")
        object.__setattr__(self, "cells", cell_set)

        rots = list(self.rotations)
        if not rots:
            raise InvalidSpecError("Part.rotations: 至少需要一个允许的旋转角")
        rot_set = set()
        for i, r in enumerate(rots):
            r = _require_int(r, f"Part.rotations[{i}]")
            if r not in _VALID_ROTATIONS:
                raise InvalidSpecError(
                    f"Part.rotations[{i}]: 非法角度 {r}, 只允许 0/90/180/270"
                )
            rot_set.add(r)
        object.__setattr__(self, "rotations", frozenset(rot_set))

        if not isinstance(self.allow_reflection, bool):
            raise TypeError(
                "Part.allow_reflection: 需要 bool, 得到 "
                f"{type(self.allow_reflection).__name__}"
            )


@dataclass(frozen=True)
class Board:
    """板材: width x height 网格 + 禁放格。"""

    width: int
    height: int
    forbidden: Iterable[Tuple[int, int]] = ()

    def __post_init__(self) -> None:
        w = _require_int(self.width, "Board.width")
        h = _require_int(self.height, "Board.height")
        if not (1 <= w <= MAX_BOARD_WIDTH):
            raise InvalidSpecError(
                f"Board.width: {w} 超出范围 [1, {MAX_BOARD_WIDTH}]"
            )
        if not (1 <= h <= MAX_BOARD_HEIGHT):
            raise InvalidSpecError(
                f"Board.height: {h} 超出范围 [1, {MAX_BOARD_HEIGHT}]"
            )
        parsed = [
            _parse_cell(c, f"Board.forbidden[{i}]")
            for i, c in enumerate(self.forbidden)
        ]
        fset = frozenset(parsed)
        if len(fset) != len(parsed):
            raise InvalidSpecError("Board.forbidden: 存在重复格子")
        for x, y in sorted(fset):
            if not (0 <= x < w and 0 <= y < h):
                raise InvalidSpecError(
                    f"Board.forbidden: 格子 ({x}, {y}) 超出板材范围"
                )
        object.__setattr__(self, "width", w)
        object.__setattr__(self, "height", h)
        object.__setattr__(self, "forbidden", fset)


class Status(Enum):
    """求解结果状态。预算耗尽与证明无解严格区分。"""

    OPTIMAL = "optimal"
    """已证明: 最小高度及全部次序目标均达到最优。"""

    INFEASIBLE = "infeasible"
    """已证明: 任何高度 (<= board.height) 都无法放下全部零件。"""

    BUDGET_EXHAUSTED = "budget_exhausted"
    """搜索预算耗尽: 报告最优已知解(可能为 None)与已证明的下界。"""


@dataclass(frozen=True)
class Placement:
    """一个零件的放置结果。

    ``x, y``: 该零件所取方向归一化形状(最小坐标为 (0,0))放置后的锚点,
    即实际占用格 = 归一化形状 + (x, y)。
    ``rotation_degrees``: 实际旋转角 (0/90/180/270)。
    ``mirrored``: 是否经过镜像。
    """

    part_index: int
    x: int
    y: int
    rotation_degrees: int
    mirrored: bool


@dataclass(frozen=True)
class PackResult:
    """求解结果。"""

    status: Status
    height: Optional[int]
    """最优(或最优已知)使用高度; 无可行解时为 None。"""

    placements: Optional[Tuple[Placement, ...]]
    """对应的放置方案(按零件下标排序); 无可行解时为 None。"""

    rotation_count: Optional[int]
    """该方案的旋转次数; 无可行解时为 None。"""

    lower_bound: int
    """已证明的下界: 任何可行解的使用高度 >= lower_bound。
    OPTIMAL 时等于 height; INFEASIBLE 时等于 board.height + 1。"""

    nodes: int
    """实际消耗的搜索节点数。"""

    budget: int
    """本次求解的搜索预算(节点数上限)。"""

    @property
    def layout(self) -> Optional[Dict[int, Tuple[int, int, int, bool]]]:
        """布局坐标字典: part_index -> (x, y, rotation_degrees, mirrored)。"""
        if self.placements is None:
            return None
        return {
            p.part_index: (p.x, p.y, p.rotation_degrees, p.mirrored)
            for p in self.placements
        }


# ---------------------------------------------------------------------------
# 切割段: 多边形零件(有理坐标直边)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolyPart:
    """一个由直线段构成的简单平面零件(含 0 个或多个孔洞)。

    专门供 :func:`cut_segments` 使用, 与网格排样用的 :class:`Part` 相互独立
    (后者描述格子零件, 没有连续轮廓)。

    - ``outer``: 外轮廓顶点(有序, 首尾不要重复), 边数 <= MAX_CONTOUR_VERTICES;
      只要求不自交、相邻边不共线退化, 顺/逆时针均可(内部自动归一化为逆时针)。
    - ``holes``: 孔洞轮廓序列, 每个同 ``outer``; 孔数 <= MAX_POLY_HOLES。
      孔洞必须完整位于外轮廓内部, 孔与孔互不相交(允许在点上相接)。
    - 全部坐标必须是精确有理数(``int`` 或 :class:`fractions.Fraction`),
      拒绝 float(包括 NaN/Infinity) —— 切割几何全部用精确有理数计算。
    - 构造时归一化: 顶点存为 ``Fraction``, 外轮廓逆时针、孔洞顺时针。
    """

    outer: Iterable[Tuple[Any, Any]]
    holes: Iterable[Iterable[Tuple[Any, Any]]] = ()

    def __post_init__(self) -> None:
        outer = _parse_contour(self.outer, "PolyPart.outer")
        if len(outer) < 3:
            raise InvalidSpecError(
                f"PolyPart.outer: 至少需要 3 个顶点, 得到 {len(outer)} 个"
            )
        if len(outer) > MAX_CONTOUR_VERTICES:
            raise InvalidSpecError(
                f"PolyPart.outer: 顶点数 {len(outer)} 超过上限 "
                f"{MAX_CONTOUR_VERTICES}"
            )
        holes_raw = list(self.holes)
        if len(holes_raw) > MAX_POLY_HOLES:
            raise InvalidSpecError(
                f"PolyPart.holes: 孔数 {len(holes_raw)} 超过上限 {MAX_POLY_HOLES}"
            )
        holes = tuple(
            _parse_contour(h, f"PolyPart.holes[{i}]")
            for i, h in enumerate(holes_raw)
        )
        for i, hc in enumerate(holes):
            if len(hc) < 3:
                raise InvalidSpecError(
                    f"PolyPart.holes[{i}]: 至少需要 3 个顶点, 得到 {len(hc)} 个"
                )
            if len(hc) > MAX_CONTOUR_VERTICES:
                raise InvalidSpecError(
                    f"PolyPart.holes[{i}]: 顶点数 {len(hc)} 超过上限 "
                    f"{MAX_CONTOUR_VERTICES}"
                )
        # 几何合法性(自交、退化、孔洞包含关系)
        _validate_simple_polygon(outer, "PolyPart.outer")
        for i, hc in enumerate(holes):
            _validate_simple_polygon(hc, f"PolyPart.holes[{i}]")
            for q in hc:
                if not _point_in_polygon(q, outer):
                    raise InvalidSpecError(
                        f"PolyPart.holes[{i}]: 孔洞顶点 {_fmt_pt(q)} 不在 "
                        "外轮廓内部"
                    )
        for i in range(len(holes)):
            for j in range(i + 1, len(holes)):
                if _polygons_overlap_interior(holes[i], holes[j]):
                    raise InvalidSpecError(
                        f"PolyPart.holes[{i}] 与 PolyPart.holes[{j}] 互相交叠"
                    )
        # 方向归一化: 外轮廓 CCW(面积为正), 孔洞 CW(面积为负)
        outer = _orient_ccw(outer)
        holes = tuple(_orient_cw(h) for h in holes)
        object.__setattr__(self, "outer", outer)
        object.__setattr__(self, "holes", holes)

    def contours(self) -> Tuple[Tuple[Tuple[Fraction, Fraction], ...], ...]:
        """全部轮廓(外轮廓在前, 已方向归一化)。"""
        return (self.outer,) + self.holes


# ---------------------------------------------------------------------------
# 有理直线段几何(全部精确运算, 不使用 float)
# ---------------------------------------------------------------------------

_Pt = Tuple[Fraction, Fraction]


def _parse_contour(vertices, path: str) -> Tuple[_Pt, ...]:
    if isinstance(vertices, (str, bytes)):
        raise TypeError(f"{path}: 需要点序列, 得到 {type(vertices).__name__}")
    pts = list(vertices)
    if not pts:
        raise InvalidSpecError(f"{path}: 轮廓为空")
    # 容忍调用方重复书写闭合点(末点 == 首点): 去掉
    if len(pts) > 1:
        first, last = pts[0], pts[-1]
        if (
            isinstance(first, (tuple, list))
            and isinstance(last, (tuple, list))
            and len(first) == 2
            and len(last) == 2
            and first[0] == last[0]
            and first[1] == last[1]
        ):
            pts = pts[:-1]
    return tuple(_parse_point(p, f"{path}[{i}]") for i, p in enumerate(pts))


def _fmt_pt(p: _Pt) -> str:
    return f"({p[0]}, {p[1]})"


def _sub(a: _Pt, b: _Pt) -> Tuple[Fraction, Fraction]:
    return (a[0] - b[0], a[1] - b[1])


def _cross(u, v) -> Fraction:
    return u[0] * v[1] - u[1] * v[0]


def _signed_area2(ring: Sequence[_Pt]) -> Fraction:
    """两倍有向面积(CCW 为正)。"""
    s = Fraction(0)
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s


def _orient_ccw(ring: Tuple[_Pt, ...]) -> Tuple[_Pt, ...]:
    return ring if _signed_area2(ring) > 0 else tuple(reversed(ring))


def _orient_cw(ring: Tuple[_Pt, ...]) -> Tuple[_Pt, ...]:
    return ring if _signed_area2(ring) < 0 else tuple(reversed(ring))


def _on_segment(p: _Pt, a: _Pt, b: _Pt) -> bool:
    """p 是否落在闭线段 [a, b] 上。"""
    if _cross(_sub(b, a), _sub(p, a)) != 0:
        return False
    return (
        min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
    )


def _segments_intersect(a: _Pt, b: _Pt, c: _Pt, d: _Pt) -> bool:
    """两条闭线段是否存在任何公共点(含端点接触与共线重叠)。"""
    r, s = _sub(b, a), _sub(d, c)
    rxs = _cross(r, s)
    qpxr = _cross(_sub(c, a), r)
    if rxs != 0:
        t = _cross(_sub(c, a), s) / rxs
        u = qpxr / rxs
        return 0 <= t <= 1 and 0 <= u <= 1
    if qpxr != 0:
        return False  # 平行不共线
    # 共线: 任一端点落在另一条上即有公共点
    return (
        _on_segment(c, a, b)
        or _on_segment(d, a, b)
        or _on_segment(a, c, d)
        or _on_segment(b, c, d)
    )


def _proper_cross(a: _Pt, b: _Pt, c: _Pt, d: _Pt) -> bool:
    """非共线且交点严格位于两条线段内部(真穿越)。"""
    r, s = _sub(b, a), _sub(d, c)
    rxs = _cross(r, s)
    if rxs == 0:
        return False
    t = _cross(_sub(c, a), s) / rxs
    u = _cross(_sub(c, a), r) / rxs
    return 0 < t < 1 and 0 < u < 1


def _collinear_overlap(a: _Pt, b: _Pt, c: _Pt, d: _Pt) -> bool:
    """两条共线线段是否有长度为正的重叠。"""
    r = _sub(b, a)
    s = _sub(d, c)
    # 先要求方向平行, 再要求 c 落在 a-b 所在直线上; 仅端点接触时
    # (如 c == a 但两方向不平行)不会被误判为共线。
    if _cross(r, s) != 0 or _cross(r, _sub(c, a)) != 0:
        return False
    denom = r[0] * r[0] + r[1] * r[1]
    if denom == 0:
        return False

    def t_of(p):
        d0 = _sub(p, a)
        return (d0[0] * r[0] + d0[1] * r[1]) / denom

    ts = sorted((t_of(c), t_of(d)))
    lo = max(Fraction(0), ts[0])
    hi = min(Fraction(1), ts[1])
    return hi > lo


def _validate_simple_polygon(ring: Tuple[_Pt, ...], path: str) -> None:
    """简单多边形校验: 无重复相邻点、无退化共线邻边、无自交。"""
    n = len(ring)
    for i in range(n):
        if ring[i] == ring[(i + 1) % n]:
            raise InvalidSpecError(
                f"{path}: 相邻顶点 {i} 与 {(i + 1) % n} 重合"
            )
    # 相邻边不得共线(折返尖刺或冗余点都视为退化)
    for i in range(n):
        u = _sub(ring[(i + 1) % n], ring[i])
        v = _sub(ring[(i + 2) % n], ring[(i + 1) % n])
        if _cross(u, v) == 0:
            raise InvalidSpecError(
                f"{path}: 顶点 {_fmt_pt(ring[(i + 1) % n])} 处相邻边共线(退化)"
            )
    # 不相邻边不得有任何公共点
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # 首尾相邻
            c, d = ring[j], ring[(j + 1) % n]
            if _segments_intersect(a, b, c, d):
                raise InvalidSpecError(
                    f"{path}: 边 ({_fmt_pt(a)}-{_fmt_pt(b)}) 与 "
                    f"({_fmt_pt(c)}-{_fmt_pt(d)}) 自交"
                )


def _point_in_polygon(p: _Pt, ring: Sequence[_Pt]) -> bool:
    """p 是否严格位于多边形内部(边界上返回 False)。半开射线法, 精确有理。"""
    x, y = p
    n = len(ring)
    for i in range(n):
        if _on_segment(p, ring[i], ring[(i + 1) % n]):
            return False
    inside = False
    for i in range(n):
        ax, ay = ring[i]
        bx, by = ring[(i + 1) % n]
        if (ay > y) != (by > y):
            x_cross = ax + (bx - ax) * (y - ay) / (by - ay)
            if x < x_cross:
                inside = not inside
    return inside


def _polygons_overlap_interior(a: Sequence[_Pt], b: Sequence[_Pt]) -> bool:
    """两个简单多边形内部是否相交(仅点接触不算)。"""
    n, m = len(a), len(b)
    for i in range(n):
        for j in range(m):
            p, q = a[i], a[(i + 1) % n]
            r, s = b[j], b[(j + 1) % m]
            if _proper_cross(p, q, r, s):
                return True
            if _collinear_overlap(p, q, r, s):
                return True
    for p in a:
        if _point_in_polygon(p, b):
            return True
    for p in b:
        if _point_in_polygon(p, a):
            return True
    return False


def _on_any_edge(p: _Pt, ring: Sequence[_Pt]) -> bool:
    n = len(ring)
    for i in range(n):
        if _on_segment(p, ring[i], ring[(i + 1) % n]):
            return True
    return False


def _region_contains_strict(p: _Pt, region: Sequence[Sequence[_Pt]]) -> bool:
    """点 p 是否严格位于"外轮廓减去孔洞"区域内部(边界上为 False)。"""
    if not _point_in_polygon(p, region[0]):
        return False
    for hole in region[1:]:
        if _on_any_edge(p, hole) or _point_in_polygon(p, hole):
            return False
    return True


def _ring_bbox(ring: Sequence[_Pt]) -> Tuple[Fraction, Fraction, Fraction, Fraction]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def _regions_overlap(
    ri: Sequence[Sequence[_Pt]], rj: Sequence[Sequence[_Pt]]
) -> bool:
    """两个"外轮廓减孔洞"填充区域是否有正面积交集。

    前置: 外轮廓已归一化为 CCW、孔洞为 CW —— 此约定下实体填充恒在每条
    有向边的 **左侧**(外轮廓内部在左; 沿孔洞 CW 行走时, 实体所在的孔外
    也在左)。允许: 沿任何轮廓边贴边、仅点接触、一个零件整体位于另一个
    零件的孔洞内(轮廓可重合但填充不相交)。
    拒绝: 边真穿越、包含、共线同侧重叠。
    """
    # 外轮廓包围盒不相交 => 一定不重叠(快速剪枝)
    ax0, ay0, ax1, ay1 = _ring_bbox(ri[0])
    bx0, by0, bx1, by1 = _ring_bbox(rj[0])
    if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
        return False

    edges_i = []
    for ring in ri:
        for i in range(len(ring)):
            edges_i.append((ring[i], ring[(i + 1) % len(ring)]))
    edges_j = []
    for ring in rj:
        for j in range(len(ring)):
            edges_j.append((ring[j], ring[(j + 1) % len(ring)]))

    for (p, q) in edges_i:
        di = _sub(q, p)
        for (u, v) in edges_j:
            dj = _sub(v, u)
            if _proper_cross(p, q, u, v):
                return True
            if _collinear_overlap(p, q, u, v):
                # 两边共线时, 实体都在各自有向边左侧: 边同向 ⇒ 实体在直线
                # 同侧 ⇒ 正面积重叠; 边反向 ⇒ 异侧 ⇒ 合法贴边(含孔内嵌件)。
                if di[0] * dj[0] + di[1] * dj[1] > 0:
                    return True

    # 包含: 任一区域的环顶点严格落在对方填充区域内 => 正面积重叠。
    # 零件整体位于对方孔洞内时, 其顶点在对方孔洞中(非填充区), 不算重叠。
    for ring in ri:
        for p in ring:
            if _region_contains_strict(p, rj):
                return True
    for ring in rj:
        for p in ring:
            if _region_contains_strict(p, ri):
                return True
    return False


# ---------------------------------------------------------------------------
# 方向生成(等形对称去重)
# ---------------------------------------------------------------------------


def _rot90(cells: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """绕原点逆时针 90 度: (x, y) -> (-y, x)。"""
    return [(-y, x) for x, y in cells]


def _reflect_v(cells: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """竖直轴镜像: (x, y) -> (-x, y)。"""
    return [(-x, y) for x, y in cells]


def oriented_cells(
    cells: Iterable[Tuple[int, int]], rotation_degrees: int, mirrored: bool
) -> Tuple[Tuple[int, int], ...]:
    """返回 ``cells`` 经(可选)镜像 + 旋转并平移归一化后的格子元组。

    先镜像(竖直轴), 再逆时针旋转 ``rotation_degrees`` 度。
    该函数与求解器内部的方向生成使用同一代码路径。
    """
    cell_set = frozenset(
        _parse_cell(c, f"oriented_cells.cells[{i}]") for i, c in enumerate(cells)
    )
    if not cell_set:
        raise InvalidSpecError("oriented_cells.cells: 至少需要 1 个格子")
    rotation_degrees = _require_int(rotation_degrees, "rotation_degrees")
    if rotation_degrees not in _VALID_ROTATIONS:
        raise InvalidSpecError(
            f"rotation_degrees: 非法角度 {rotation_degrees}, 只允许 0/90/180/270"
        )
    if not isinstance(mirrored, bool):
        raise TypeError(f"mirrored: 需要 bool, 得到 {type(mirrored).__name__}")
    cur: List[Tuple[int, int]] = sorted(cell_set)
    if mirrored:
        cur = _reflect_v(cur)
    for _ in range(rotation_degrees // 90):
        cur = _rot90(cur)
    return _normalize_cells(cur)


@dataclass(frozen=True)
class _Variant:
    """一个去重后的方向: 归一化格子 + 代表 (rotation, mirrored) + 尺寸。"""

    cells: Tuple[Tuple[int, int], ...]
    rotation: int
    mirrored: bool
    w: int
    h: int


def _variants_for(part: Part, allow_reflection: bool) -> Tuple[_Variant, ...]:
    """生成零件的全部不同方向(等形同余去重)。

    去重: 由对称性产生的相同格子集合只保留代表元
    (mirrored=False 优先, 再按旋转角升序)。
    """
    base = sorted(part.cells)
    rots = set(part.rotations)
    seen: Dict[Tuple[Tuple[int, int], ...], Tuple[int, bool]] = {}
    use_mirror = allow_reflection and part.allow_reflection
    for mirrored in ([False, True] if use_mirror else [False]):
        cur = _reflect_v(base) if mirrored else list(base)
        deg = 0
        for _ in range(4):
            if deg in rots:
                norm = _normalize_cells(cur)
                if norm not in seen:
                    seen[norm] = (deg, mirrored)
            cur = _rot90(cur)
            deg += 90
    variants = [
        _Variant(norm, deg, mir, max(x for x, _ in norm) + 1, max(y for _, y in norm) + 1)
        for norm, (deg, mir) in seen.items()
    ]
    # 排序确定性: 恒等方向优先, 便于"最少旋转"目标尽早找到好解
    variants.sort(key=lambda v: (v.mirrored, v.rotation))
    return tuple(variants)


# ---------------------------------------------------------------------------
# 位集工具
# ---------------------------------------------------------------------------

_EDGE_ROWS = MAX_BOARD_HEIGHT + 2  # 膨胀可能超出板一行, 边缘掩码多留行


def _build_edge_masks():
    not_left: Dict[int, int] = {}  # x >= 1 的位
    not_right: Dict[int, int] = {}  # x <= w-2 的位
    for w in range(1, MAX_BOARD_WIDTH + 1):
        nl = nr = 0
        for y in range(_EDGE_ROWS):
            for x in range(w):
                if x >= 1:
                    nl |= 1 << (y * w + x)
                if x <= w - 2:
                    nr |= 1 << (y * w + x)
        not_left[w] = nl
        not_right[w] = nr
    return not_left, not_right


_NOT_LEFT_EDGE, _NOT_RIGHT_EDGE = _build_edge_masks()


def _dilate(mask: int, w: int) -> int:
    """对 w 列位图做 Chebyshev 一格膨胀(跨列进位按板材边界裁剪)。

    ``mask >> 1`` 的合法结果位不会落在 x == w-1 (那是上一行 x=0 绕回);
    ``mask << 1`` 的合法结果位不会落在 x == 0。
    """
    left = (mask >> 1) & _NOT_RIGHT_EDGE[w]
    right = (mask << 1) & _NOT_LEFT_EDGE[w]
    spread = mask | left | right
    return spread | (spread << w) | (spread >> w)


def _shape_mask(cells: Iterable[Tuple[int, int]], w: int) -> int:
    m = 0
    for x, y in cells:
        m |= 1 << (y * w + x)
    return m


@dataclass(frozen=True)
class _Cand:
    """一个候选放置。"""

    x: int
    y: int
    vi: int  # 方向下标(在所属组的 _Variant 列表中)
    mask: int  # 实际占用位掩码
    dil: int  # Chebyshev 一格膨胀位掩码(板外裁剪)


# ---------------------------------------------------------------------------
# 回溯搜索
# ---------------------------------------------------------------------------


class _BudgetExhausted(Exception):
    pass


class _GreedyDone(Exception):
    """贪心模式找到首个可行叶节点, 携带放置下标元组。"""

    def __init__(self, assignment: Tuple[int, ...]):
        super().__init__()
        self.assignment = assignment


class _Search:
    """单次(固定高度 h)DFS。预算通过可变列表 counter 与全局共享。"""

    def __init__(
        self,
        variants: List[Tuple[_Variant, ...]],
        cands: List[List[_Cand]],
        groups: List[List[int]],
        counter: List[int],
        budget: int,
        enumerate_all: bool,
    ):
        self.variants = variants
        self.cands = cands
        self.groups = groups
        self.counter = counter
        self.budget = budget
        self.enumerate_all = enumerate_all
        n_parts = sum(len(g) for g in groups)
        # 每个零件的放置 = 候选下标; -1 表示未放置
        self.assign: List[int] = [-1] * n_parts
        self.last_in_group: List[int] = [-1] * len(groups)
        self.best_key: Optional[Tuple] = None
        self.best_assign: Optional[Tuple[int, ...]] = None

    # -- 主递归 ------------------------------------------------------------
    def dfs(self, placed: int, mask_u: int, dil_u: int, rot_cost: int) -> None:
        self.counter[0] += 1
        if self.counter[0] > self.budget:
            raise _BudgetExhausted
        if placed == len(self.assign):
            self._on_leaf(rot_cost)
            return
        if (
            self.enumerate_all
            and self.best_key is not None
            and rot_cost > self.best_key[0]
        ):
            return  # 旋转次数已不可能更优
        # MRV: 选择可行候选最少的未完成组(扫描带截断, 只需知道是否更少)
        g_best = -1
        target_best = -1
        opts_best: Optional[List[int]] = None
        for g, members in enumerate(self.groups):
            target = -1
            for p in members:
                if self.assign[p] < 0:
                    target = p
                    break
            if target < 0:
                continue
            cap = len(opts_best) if opts_best is not None else None
            opts = self._options(g, mask_u, dil_u, cap)
            if not opts:
                return
            if opts_best is None or len(opts) < len(opts_best):
                opts_best = opts
                g_best = g
                target_best = target
                if len(opts) == 1:
                    break
        assert opts_best is not None
        for ci in opts_best:
            cand = self.cands[g_best][ci]
            self.assign[target_best] = ci
            prev_last = self.last_in_group[g_best]
            self.last_in_group[g_best] = ci
            self.dfs(
                placed + 1,
                mask_u | cand.mask,
                dil_u | cand.dil,
                rot_cost + (0 if self.variants[g_best][cand.vi].rotation == 0 else 1),
            )
            self.assign[target_best] = -1
            self.last_in_group[g_best] = prev_last

    def _options(
        self, g: int, mask_u: int, dil_u: int, cap: Optional[int] = None
    ) -> List[int]:
        """组 g 当前的可行候选下标(含同组对称破缺: 下标单调不减)。

        候选在生成时已排除禁放格, 这里只检查零件间冲突。
        cap: 收集到 cap 个即可提前返回(调用方只关心是否比已知最少更少)。
        """
        start = self.last_in_group[g] + 1
        out = []
        for ci in range(start, len(self.cands[g])):
            cand = self.cands[g][ci]
            if cand.mask & dil_u:  # 与已放零件的膨胀区相交
                continue
            if cand.dil & mask_u:  # 膨胀区覆盖已放零件
                continue
            out.append(ci)
            if cap is not None and len(out) >= cap:
                break
        return out

    def _on_leaf(self, rot_cost: int) -> None:
        if not self.enumerate_all:
            raise _GreedyDone(tuple(self.assign))
        key = (rot_cost, self._layout_key())
        if self.best_key is None or key < self.best_key:
            self.best_key = key
            self.best_assign = tuple(self.assign)

    def _layout_key(self) -> Tuple:
        """按零件下标顺序的 ((x, y, (rotation, mirrored)), ...) 元组。

        同组(完全相同)零件可互换: 组内条目按字典序排序后填入递增的零件
        下标槽位(对称破缺只保留一种组内排列, 规范化保证不丢最优);
        最终键严格按零件下标顺序发射(同组成员的下标可能不连续)。
        """
        entries_by_part: Dict[int, Tuple] = {}
        for g, members in enumerate(self.groups):
            entries = []
            for p in members:
                cand = self.cands[g][self.assign[p]]
                var = self.variants[g][cand.vi]
                entries.append((cand.x, cand.y, (var.rotation, var.mirrored)))
            entries.sort()
            for p, entry in zip(members, entries):
                entries_by_part[p] = entry
        return tuple(entries_by_part[p] for p in range(len(self.assign)))


# ---------------------------------------------------------------------------
# 候选生成
# ---------------------------------------------------------------------------


def _build_candidates(
    variants: Tuple[_Variant, ...], board: Board, h: int, forbidden_h: int
) -> List[_Cand]:
    """一组相同零件在给定高度 h 下的全部合法候选放置。"""
    w = board.width
    out: List[_Cand] = []
    for vi, var in enumerate(variants):
        if var.w > w or var.h > h:
            continue
        base_mask = _shape_mask(var.cells, w)
        for y in range(h - var.h + 1):
            for x in range(w - var.w + 1):
                mask = base_mask << (y * w + x)
                if mask & forbidden_h:
                    continue
                dil = _dilate(mask, w)
                out.append(_Cand(x, y, vi, mask, dil))
    # 底部优先的顺序, 让贪心快速得到低高度的可行解
    # (枚举阶段做完整遍历, 顺序不影响最终字典序最优性)
    out.sort(key=lambda c: (c.y, c.x, c.vi))
    return out


def _forbidden_mask(board: Board, h: int) -> int:
    """禁放格在底部 h 行内的位掩码。"""
    m = 0
    for x, y in board.forbidden:
        if y < h:
            m |= 1 << (y * board.width + x)
    return m


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def pack(
    board: Board,
    parts: Sequence[Part],
    budget: int = DEFAULT_BUDGET,
    allow_reflection: bool = True,
) -> PackResult:
    """求解无重叠排样。

    参数
    ----
    board: Board 实例。
    parts: Part 序列, 长度 1..MAX_PARTS。
    budget: 搜索预算(DFS 节点数, 正整数)。所有高度的搜索共享同一预算。
    allow_reflection: 全局镜像开关; False 时所有零件禁用镜像。

    返回 PackResult; 不会静默修正非法输入(类型错误抛 TypeError,
    取值域违规抛 InvalidSpecError)。
    """
    if not isinstance(board, Board):
        raise TypeError(f"board: 需要 Board, 得到 {type(board).__name__}")
    if isinstance(parts, (str, bytes)):
        raise TypeError("parts: 需要 Part 序列")
    try:
        parts = list(parts)
    except TypeError:
        raise TypeError("parts: 需要 Part 序列") from None
    for i, p in enumerate(parts):
        if not isinstance(p, Part):
            raise TypeError(f"parts[{i}]: 需要 Part, 得到 {type(p).__name__}")
    if not (1 <= len(parts) <= MAX_PARTS):
        raise InvalidSpecError(
            f"parts: 零件数 {len(parts)} 超出范围 [1, {MAX_PARTS}]"
        )
    budget = _require_int(budget, "budget")
    if budget < 1:
        raise InvalidSpecError(f"budget: 需要 >= 1, 得到 {budget}")
    if not isinstance(allow_reflection, bool):
        raise TypeError(
            "allow_reflection: 需要 bool, 得到 "
            f"{type(allow_reflection).__name__}"
        )

    counter = [0]
    w, H = board.width, board.height
    total_area = sum(len(p.cells) for p in parts)

    # 每个零件的方向(等形去重后)
    variants = [_variants_for(p, allow_reflection) for p in parts]

    # 完全相同的零件(允许方向集合一致, 含代表旋转/镜像元数据)归为一组,
    # 做对称破缺。注意: 形状相同但允许旋转集不同的零件不可互换
    # (旋转次数目标不同), 不能同组。
    groups: List[List[int]] = []
    sig_to_group: Dict[Tuple, int] = {}
    for i, v in enumerate(variants):
        sig = tuple((vv.cells, vv.rotation, vv.mirrored) for vv in v)
        g = sig_to_group.get(sig)
        if g is None:
            g = len(groups)
            sig_to_group[sig] = g
            groups.append([])
        groups[g].append(i)
    group_variants = [variants[members[0]] for members in groups]

    # 下界: 面积下界与"每零件最小包围盒高度"下界的较大者
    lb_area = -(-total_area // w)  # ceil
    lb_part = max(min(v.h for v in vs) for vs in variants)
    lb = max(1, lb_area, lb_part)

    def finish(status, height, placements, rot, lower) -> PackResult:
        return PackResult(
            status=status,
            height=height,
            placements=placements,
            rotation_count=rot,
            lower_bound=lower,
            nodes=counter[0],
            budget=budget,
        )

    def make_placements(cands_per_group, assignment) -> Tuple[Placement, ...]:
        # 同组零件可互换: 组内放置按 (x, y, (rot, mirrored)) 排序后填入
        # 递增的零件下标, 使报告的布局为字典序最小排列(与 _layout_key 一致)
        out = []
        for g, members in enumerate(groups):
            entries = []
            for p in members:
                cand = cands_per_group[g][assignment[p]]
                var = group_variants[g][cand.vi]
                entries.append((cand.x, cand.y, var.rotation, var.mirrored))
            entries.sort(key=lambda e: (e[0], e[1], (e[2], e[3])))
            for p, (x, y, rot, mir) in zip(members, entries):
                out.append(Placement(p, x, y, rot, mir))
        out.sort(key=lambda pl: pl.part_index)
        return tuple(out)

    def rot_of(placements: Tuple[Placement, ...]) -> int:
        return sum(1 for pl in placements if pl.rotation_degrees != 0)

    def used_height(cands_per_group, assignment) -> int:
        uh = 1
        for g, members in enumerate(groups):
            for p in members:
                cand = cands_per_group[g][assignment[p]]
                var = group_variants[g][cand.vi]
                uh = max(uh, cand.y + var.h)
        return uh

    # 可行性关于高度单调: 高度 h 可行 => 任何更高高度也可行。
    # -- 阶段 1: 在整板高度 H 下贪心取任一可行解(首叶即停) -------------------
    try:
        forb_H = _forbidden_mask(board, H)
        cands_g = [
            _build_candidates(gv, board, H, forb_H) for gv in group_variants
        ]
        if any(not c for c in cands_g):
            # 某零件在整板高度下都无合法放置: 整体不可行
            return finish(Status.INFEASIBLE, None, None, None, H + 1)
        s = _Search(
            group_variants, cands_g, groups, counter, budget, enumerate_all=False
        )
        incumbent_assignment: Optional[Tuple[int, ...]] = None
        try:
            s.dfs(0, 0, 0, 0)
        except _GreedyDone as gd:
            incumbent_assignment = gd.assignment
        if incumbent_assignment is None:
            # 整板高度下完整 DFS 无解 => 任何高度都不可行(单调性)
            return finish(Status.INFEASIBLE, None, None, None, H + 1)
        best_pl = make_placements(cands_g, incumbent_assignment)
        best_h = used_height(cands_g, incumbent_assignment)  # 实际使用高度
        best_enumerated = False  # best_pl 是否已来自完整枚举
    except _BudgetExhausted:
        return finish(Status.BUDGET_EXHAUSTED, None, None, None, lb)

    # -- 阶段 2: 自 lb 起逐高度完整枚举, 直到达到当前最优高度 -----------------
    try:
        h = lb
        while h < best_h:
            if h * w < total_area:
                h += 1
                continue  # 面积下界证明该高度不可行
            forb_h = _forbidden_mask(board, h)
            cands_h = [
                _build_candidates(gv, board, h, forb_h) for gv in group_variants
            ]
            if any(not c for c in cands_h):
                h += 1
                continue  # 某零件在此高度无任何合法放置: 不可行
            s = _Search(
                group_variants, cands_h, groups, counter, budget,
                enumerate_all=True,
            )
            s.dfs(0, 0, 0, 0)
            if s.best_assign is not None:
                best_h = h
                best_pl = make_placements(cands_h, s.best_assign)
                best_enumerated = True
            h += 1
        if not best_enumerated:
            # 当前最优高度来自贪心: 在该高度补一次完整枚举以优化次序目标
            forb_h = _forbidden_mask(board, best_h)
            cands_h = [
                _build_candidates(gv, board, best_h, forb_h)
                for gv in group_variants
            ]
            s = _Search(
                group_variants, cands_h, groups, counter, budget,
                enumerate_all=True,
            )
            s.dfs(0, 0, 0, 0)
            if s.best_assign is not None:
                best_pl = make_placements(cands_h, s.best_assign)
    except _BudgetExhausted:
        return finish(
            Status.BUDGET_EXHAUSTED, best_h, best_pl, rot_of(best_pl), lb
        )
    return finish(Status.OPTIMAL, best_h, best_pl, rot_of(best_pl), best_h)


# ---------------------------------------------------------------------------
# 校验与渲染(公开工具)
# ---------------------------------------------------------------------------


def verify_layout(
    board: Board,
    parts: Sequence[Part],
    placements: Sequence[Placement],
    height: int,
) -> List[str]:
    """校验一个布局是否满足全部约束; 返回问题描述列表(空列表 = 合法)。

    检查: 零件数匹配、方向被允许、板内、禁放格、互不重叠、
    零件间 Chebyshev 一格边距、使用高度 <= height。
    """
    problems: List[str] = []
    height = _require_int(height, "height")
    if not (1 <= height <= board.height):
        problems.append(f"height {height} 超出范围 [1, {board.height}]")
        return problems
    parts = list(parts)
    placements = list(placements)
    if len(placements) != len(parts):
        problems.append(
            f"placements 数量 {len(placements)} 与零件数 {len(parts)} 不一致"
        )
        return problems
    seen_idx = set()
    masks: List[int] = []
    dils: List[int] = []
    for pl in placements:
        i = pl.part_index
        if not isinstance(i, int) or not (0 <= i < len(parts)):
            problems.append(f"非法零件下标 {i!r}")
            continue
        if i in seen_idx:
            problems.append(f"零件 {i} 被重复放置")
            continue
        seen_idx.add(i)
        part = parts[i]
        try:
            shape = oriented_cells(part.cells, pl.rotation_degrees, pl.mirrored)
            px = _require_int(pl.x, f"placements[{i}].x")
            py = _require_int(pl.y, f"placements[{i}].y")
        except (TypeError, InvalidSpecError) as e:
            problems.append(f"零件 {i}: 放置非法 ({e})")
            masks.append(0)
            dils.append(0)
            continue
        if pl.rotation_degrees not in part.rotations:
            problems.append(
                f"零件 {i}: 旋转角 {pl.rotation_degrees} 不在允许集合内"
            )
        if pl.mirrored and not part.allow_reflection:
            problems.append(f"零件 {i}: 使用了镜像但该零件禁用镜像")
        cells = [(x + px, y + py) for x, y in shape]
        in_range = True
        for x, y in cells:
            if not (0 <= x < board.width and 0 <= y < height):
                problems.append(f"零件 {i}: 格子 ({x}, {y}) 超出板/高度范围")
                in_range = False
            elif (x, y) in board.forbidden:
                problems.append(f"零件 {i}: 格子 ({x}, {y}) 覆盖禁放格")
        if in_range:
            m = _shape_mask(cells, board.width)
            masks.append(m)
            dils.append(_dilate(m, board.width))
        else:
            masks.append(0)
            dils.append(0)
    union = 0
    for i, m in enumerate(masks):
        if m & union:
            problems.append(f"零件 {i} 与其他零件重叠")
        union |= m
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if masks[i] & dils[j] or masks[j] & dils[i]:
                problems.append(f"零件 {i} 与零件 {j} 之间边距不足一格")
    return problems


def render_layout(
    board: Board,
    parts: Sequence[Part],
    placements: Sequence[Placement],
    height: int,
) -> str:
    """把布局渲染为字符画(顶行在上)。

    零件用标签字符表示(0-9, A-Z, a-z 循环), 禁放格 '#', 空白 '.'。
    """
    height = _require_int(height, "height")
    if not (1 <= height <= board.height):
        raise InvalidSpecError(f"height: {height} 超出范围 [1, {board.height}]")
    labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    grid = [["." for _ in range(board.width)] for _ in range(height)]
    for x, y in board.forbidden:
        if y < height:
            grid[y][x] = "#"
    for pl in placements:
        shape = oriented_cells(
            parts[pl.part_index].cells, pl.rotation_degrees, pl.mirrored
        )
        ch = labels[pl.part_index % len(labels)]
        for x, y in shape:
            gx, gy = x + pl.x, y + pl.y
            if 0 <= gx < board.width and 0 <= gy < height:
                grid[gy][gx] = ch
    return "\n".join("".join(row) for row in reversed(grid))


# ===========================================================================
# 切割段生成
# ===========================================================================
#
# 输入: 一个已验证合法的排样结果 —— 若干 :class:`PolyPart` 零件的刚体
# 放置(平移; 旋转/镜像限定于网格零件, 多边形零件目前只支持平移放置,
# 与题面"先限有理坐标直边、固定端点排序"的范围一致)。
#
# 输出: 一组切割直线段。两个零件的轮廓边若 **几何完全重合且切缝补偿
# (kerf)相同**, 则只生成一段切割(走一刀), 并在段上保留全部相关零件
# ID; 部分重合的边先在交点处切分, 只有真正重合的子段才合并。
#
# 同一零件的外轮廓与孔洞轮廓永远不会几何重合(孔洞在内部), 因此内外
# 轮廓之间不存在可合并边; 孔洞与孔洞、孔洞与外轮廓之间同理。
#
# 端点排序固定: 每段以 (x, y) 字典序较小的端点为起点, 段集合按
# (起点, 终点, kerf) 字典序排列, 保证输出完全确定、可复现。


@dataclass(frozen=True)
class CutSegment:
    """一段最终切割直线(端点为精确有理坐标)。

    ``p1``: 起点, ``p2``: 终点(固定排序: ``p1 < p2``)。
    ``parts``: 这一刀涉及的全部零件下标(去重、升序)。共边只切一次时
      这里会有多个零件; 非共边只有一个。
    ``kerf``: 该段的切缝补偿宽度(Fraction)。
    ``shared``: 是否为共边(>1 个零件)的单次切割。
    """

    p1: Tuple[Fraction, Fraction]
    p2: Tuple[Fraction, Fraction]
    parts: Tuple[int, ...]
    kerf: Fraction
    shared: bool


@dataclass(frozen=True)
class CutPlan:
    """切割段生成结果。"""

    segments: Tuple[CutSegment, ...]
    """全部切割段(固定排序)。"""

    part_count: int
    """参与的零件数。"""

    raw_edge_count: int
    """合并前的轮廓边总数(含外轮廓与孔洞)。"""

    shared_count: int
    """由共边合并产生的段数(shared=True 的段数)。"""

    def __iter__(self):
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)


# ---------------------------------------------------------------------------
# 多边形放置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolyPlacement:
    """一个 PolyPart 的放置: 平移 (tx, ty)(只支持平移)。

    轮廓上的点 p 放置后为 (p.x + tx, p.y + ty)。
    """

    part_index: int
    tx: Any
    ty: Any


# ---------------------------------------------------------------------------
# 共边切分与合并(精确有理)
# ---------------------------------------------------------------------------


def _edge_interior_split_points(
    a: _Pt, b: _Pt, c: _Pt, d: _Pt
) -> List[_Pt]:
    """返回应在边 a-b 上插入的切分点: 与边 c-d 的交点(含落在 a-b 内部的端点)。

    用于处理"部分重合不能错误去重": 任何落在 a-b 内部的交点/c-d 端点
    都会成为新的顶点, 使后续按原子段合并时只有真正重合的部分被合并。
    """
    pts: List[_Pt] = []
    r, s = _sub(b, a), _sub(d, c)
    rxs = _cross(r, s)
    if rxs != 0:
        t = _cross(_sub(c, a), s) / rxs
        u = _cross(_sub(c, a), r) / rxs
        if 0 < t < 1 and 0 <= u <= 1:
            pts.append((a[0] + t * r[0], a[1] + t * r[1]))
        return pts
    # 共线: 把 c, d 中落在 a-b 内部的点作为切分点
    for p in (c, d):
        if _on_segment(p, a, b) and p != a and p != b:
            pts.append(p)
    return pts


def _cut_plan_for_parts(
    polygons: Sequence[Tuple[Tuple[_Pt, ...], ...]],
    translations: Sequence[Tuple[Fraction, Fraction]],
    kerfs: Sequence[Fraction],
) -> CutPlan:
    """核心合并算法(输入已完成类型/取值校验与放置合法性校验)。

    polygons[i]: 零件 i 的全部轮廓(外轮廓 + 孔洞, 已方向归一化)。
    """
    # 1) 生成全部"原子边": (p1, p2) 端点已按字典序固定, 附带零件与 kerf。
    #    key = (p1, p2); value = 每个来源一条: (part_index, kerf)
    raw_total = 0
    edge_sources: Dict[Tuple[_Pt, _Pt], List[Tuple[int, Fraction]]] = {}

    def add_edge(u: _Pt, v: _Pt, idx: int, kerf: Fraction) -> None:
        key = (u, v) if u <= v else (v, u)
        edge_sources.setdefault(key, []).append((idx, kerf))

    for i, (contours, (tx, ty)) in enumerate(zip(polygons, translations)):
        for ring in contours:
            n = len(ring)
            moved = tuple((p[0] + tx, p[1] + ty) for p in ring)
            raw_total += n
            for k in range(n):
                add_edge(moved[k], moved[(k + 1) % n], i, kerfs[i])

    # 2) 处理部分重合: 对每条边, 用其余所有边求落在其内部的交点/端点,
    #    在这些点处把边切分成原子段。切分只沿一条边做, 另一条边也会在
    #    自己的切分轮次中被同样处理, 因此重合部分最终对齐。
    keys = list(edge_sources.keys())
    # 每条原子边的切分点集合(不含端点)
    splits: Dict[Tuple[_Pt, _Pt], set] = {k: set() for k in keys}
    # 快速跳过: 只有包围盒相交的边才需要求交
    for x in range(len(keys)):
        a, b = keys[x]
        for y in range(x + 1, len(keys)):
            c, d = keys[y]
            if (
                max(min(a[0], b[0]), min(c[0], d[0]))
                > min(max(a[0], b[0]), max(c[0], d[0]))
                or max(min(a[1], b[1]), min(c[1], d[1]))
                > min(max(a[1], b[1]), max(c[1], d[1]))
            ):
                continue
            for p in _edge_interior_split_points(a, b, c, d):
                splits[keys[x]].add(p)
            for p in _edge_interior_split_points(c, d, a, b):
                splits[keys[y]].add(p)

    # 3) 把每条边切成原子段, 累积 (原子段 -> 来源列表)。
    atom_sources: Dict[Tuple[_Pt, _Pt], List[Tuple[int, Fraction]]] = {}

    def emit_atom(u: _Pt, v: _Pt, src: Tuple[int, Fraction]) -> None:
        key = (u, v) if u <= v else (v, u)
        atom_sources.setdefault(key, []).append(src)

    for key, srcs in edge_sources.items():
        a, b = key
        sp = sorted(splits.get(key, ()))
        chain = [a] + sp + [b]
        for src in srcs:
            for k in range(len(chain) - 1):
                if chain[k] != chain[k + 1]:
                    emit_atom(chain[k], chain[k + 1], src)

    # 4) 合并: 同一原子段只在 kerf 相同时合并为一刀;
    #    kerf 不同则分别成段(切缝补偿不同, 不能只切一次)。
    segments: List[CutSegment] = []
    for key, srcs in atom_sources.items():
        groups: Dict[Fraction, List[int]] = {}
        for idx, kerf in srcs:
            groups.setdefault(kerf, []).append(idx)
        p1, p2 = key
        for kerf, idxs in groups.items():
            uniq = tuple(sorted(set(idxs)))
            segments.append(
                CutSegment(
                    p1=p1, p2=p2, parts=uniq, kerf=kerf, shared=len(uniq) > 1
                )
            )

    segments = _merge_collinear_chains(segments)
    segments.sort(key=lambda seg: (seg.p1, seg.p2, seg.kerf))
    shared_count = sum(1 for seg in segments if seg.shared)
    return CutPlan(
        segments=tuple(segments),
        part_count=len(polygons),
        raw_edge_count=raw_total,
        shared_count=shared_count,
    )


def _line_id(p1: _Pt, p2: _Pt) -> Tuple[int, int, Fraction, Fraction]:
    """线段支撑直线的规范化无向标识: (ux, uy, c)。

    (ux, uy) 为约去公因子的整数方向(第一个非零分量取正);
    c = uy*x - ux*y 为直线常数(共线点相同)。
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # 有理方向 -> 原始整数方向 -> 约去 gcd
    scale = math.lcm(dx.denominator, dy.denominator)
    ix, iy = int(dx * scale), int(dy * scale)
    g = math.gcd(abs(ix), abs(iy))
    ux, uy = ix // g, iy // g
    if ux < 0 or (ux == 0 and uy < 0):
        ux, uy = -ux, -uy
    c = uy * p1[0] - ux * p1[1]
    return (ux, uy, c)


def _merge_collinear_chains(
    segments: List[CutSegment],
) -> List[CutSegment]:
    """把同一直线上、端点相接且属性一致的相邻段合并为一整段。

    仅当连接点处 **没有其他方向的切割段** 时才合并(角点/T 形接点保留);
    且两段的 parts 与 kerf 必须完全相同(否则段上属性不一致, 不能合并)。
    这样格子轮廓在共线格点处的多余顶点被消除, 整条共边成为一刀,
    而部分重合/丁字接的分界点不会丢失。
    """
    if not segments:
        return segments
    # 每个端点上出现的不同支撑线(用于判断该点是否仅在这一条直线上)
    point_lines: Dict[_Pt, set] = {}
    line_segs: Dict[Any, List[CutSegment]] = {}
    seg_lid: Dict[int, Any] = {}
    for k, seg in enumerate(segments):
        lid = _line_id(seg.p1, seg.p2)
        seg_lid[k] = lid
        line_segs.setdefault(lid, []).append(seg)
        point_lines.setdefault(seg.p1, set()).add(lid)
        point_lines.setdefault(seg.p2, set()).add(lid)

    out: List[CutSegment] = []
    for lid, segs in line_segs.items():
        ux, uy, _c = lid

        def t_of(p: _Pt) -> Fraction:
            return p[0] * ux + p[1] * uy

        # 按 (parts, kerf) 相同才可链接
        groups: Dict[Tuple[Tuple[int, ...], Fraction], List[CutSegment]] = {}
        for seg in segs:
            groups.setdefault((seg.parts, seg.kerf), []).append(seg)
        for (parts_id, kerf), group in groups.items():
            # t -> 该 t 对应的实际点(同一直线上 t 唯一确定点)
            t_point: Dict[Fraction, _Pt] = {}
            intervals = []
            for s in group:
                t_point[t_of(s.p1)] = s.p1
                t_point[t_of(s.p2)] = s.p2
                intervals.append((min(t_of(s.p1), t_of(s.p2)),
                                  max(t_of(s.p1), t_of(s.p2))))
            intervals.sort()
            merged: List[Tuple[Fraction, Fraction]] = []
            for a, b in intervals:
                if merged and a == merged[-1][1] and len(point_lines[t_point[a]]) == 1:
                    merged[-1] = (merged[-1][0], b)  # 纯直线内部点: 合并
                else:
                    merged.append((a, b))
            for a, b in merged:
                pa, pb = t_point[a], t_point[b]
                out.append(
                    CutSegment(
                        p1=pa if pa <= pb else pb,
                        p2=pb if pa <= pb else pa,
                        parts=parts_id,
                        kerf=kerf,
                        shared=len(parts_id) > 1,
                    )
                )
    return out


# ---------------------------------------------------------------------------
# 格子零件 -> 直线段轮廓(供排样结果生成切割段)
# ---------------------------------------------------------------------------

# 方向编码: 0=东(E), 1=北(N), 2=西(W), 3=南(S); 向量与左转(CCW)。
_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _grid_contours(cells: Iterable[Tuple[int, int]]) -> Tuple[Tuple[_Pt, ...], ...]:
    """把 4-连通格子集合的边界提取为有序轮廓环。

    每条边界边取"实体填充在有向边左侧"的方向: 外轮廓因此为 CCW,
    孔洞轮廓为 CW, 与 :class:`PolyPart` 的归一化约定一致, 可直接进入
    同一套共边切分/合并与重叠判定。在鞍点(度 4 的格子顶点)处按最左
    转弯选边, 正确分离内外环而不串环。
    """
    cell_set = frozenset(cells)
    # out[(起点, 方向)] = 终点; 方向见 _DIRS
    used = set()
    out: Dict[Tuple[_Pt, int], _Pt] = {}
    for (x, y) in cell_set:
        corners = (
            (Fraction(x), Fraction(y)),
            (Fraction(x + 1), Fraction(y)),
            (Fraction(x + 1), Fraction(y + 1)),
            (Fraction(x), Fraction(y + 1)),
        )
        # 邻居缺失则该侧为边界; 方向使格子(实体)位于左侧
        if (x, y - 1) not in cell_set:  # 南: 朝东, 左=北=实体
            out[(corners[0], 0)] = corners[1]
        if (x + 1, y) not in cell_set:  # 东: 朝北, 左=西=实体
            out[(corners[1], 1)] = corners[2]
        if (x, y + 1) not in cell_set:  # 北: 朝西, 左=南=实体
            out[(corners[2], 2)] = corners[3]
        if (x - 1, y) not in cell_set:  # 西: 朝南, 左=东=实体
            out[(corners[3], 3)] = corners[0]

    rings: List[Tuple[_Pt, ...]] = []
    remaining = set(out.keys())
    while remaining:
        first_key = next(iter(remaining))
        start = first_key[0]
        cur = start
        direction = first_key[1]
        ring_pts = [cur]
        while True:
            key = (cur, direction)
            used.add(key)
            remaining.discard(key)
            dx, dy = _DIRS[direction]
            nxt = (cur[0] + dx, cur[1] + dy)
            # 到达 nxt, 按"最左转弯"选择出方向以保持填充在左:
            # 左转90 -> 直行 -> 右转90 -> 掉头
            chosen = -1
            for turn in (1, 0, -1, 2):
                nd = (direction + turn) % 4
                if (nxt, nd) in out and (nxt, nd) not in used:
                    chosen = nd
                    break
            if chosen < 0 or nxt == start:
                break  # 回到起点闭环
            ring_pts.append(nxt)
            cur = nxt
            direction = chosen
        rings.append(tuple(ring_pts))
    # 外轮廓(面积最大者)在前
    rings.sort(key=lambda r: -abs(_signed_area2(r)))
    return tuple(rings)


# ---------------------------------------------------------------------------
# 放置合法性校验
# ---------------------------------------------------------------------------


def _check_poly_placements(
    parts: Sequence[PolyPart],
    placements: Sequence[PolyPlacement],
    translations: List[Tuple[Fraction, Fraction]],
) -> None:
    """校验放置结果合法: 下标完整唯一、轮廓在板内、零件互不重叠(含边距)。

    排样题面要求对"已有合法排样结果"生成切割段; 这里独立复核, 不合法即
    报错且不产生任何结果(无半更新状态 —— 结果只通过返回值给出)。
    """
    n = len(parts)
    if len(placements) != n:
        raise InvalidSpecError(
            f"placements: 数量 {len(placements)} 与零件数 {n} 不一致"
        )
    seen = set()
    for k, pl in enumerate(placements):
        i = pl.part_index
        if not isinstance(i, int) or isinstance(i, bool):
            raise TypeError(f"placements[{k}].part_index: 需要 int")
        if not (0 <= i < n):
            raise InvalidSpecError(
                f"placements[{k}].part_index: {i} 超出零件下标范围 [0, {n - 1}]"
            )
        if i in seen:
            raise InvalidSpecError(f"placements: 零件 {i} 被重复放置")
        seen.add(i)

    placed_polys = [
        (
            tuple(
                tuple((p[0] + translations[k][0], p[1] + translations[k][1]) for p in ring)
                for ring in parts[pl.part_index].contours()
            )
        )
        for k, pl in enumerate(placements)
    ]

    # 板内: 所有顶点(平移后)非负。题目板材为第一象限整数网格。
    # 放宽到只要求非负(不强制整数板宽), 因为切割几何本身是有理的;
    # 具体上界由调用方的排样结果保证, 这里检查坐标非负即可覆盖越界符号。
    for k, polys in enumerate(placed_polys):
        for ring in polys:
            for p in ring:
                if p[0] < 0 or p[1] < 0:
                    raise InvalidSpecError(
                        f"零件 {placements[k].part_index}: 放置后顶点 "
                        f"({p[0]}, {p[1]}) 越出板材(负坐标)"
                    )

    # 互不重叠: 比较两个"外轮廓减孔洞"填充区域, 允许沿边界贴边(共边切割
    # 要求这种放置合法), 也允许一个零件整体落在另一个零件的孔洞内
    # (此时轮廓可能重合, 但填充区域不相交)。
    for x in range(n):
        for y in range(x + 1, n):
            if _regions_overlap(placed_polys[x], placed_polys[y]):
                raise InvalidSpecError(
                    f"零件 {placements[x].part_index} 与零件 "
                    f"{placements[y].part_index} 的排样区域互相重叠"
                )


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------


def cut_segments(
    parts: Sequence[PolyPart],
    placements: Sequence[PolyPlacement],
    kerf: Any = 0,
    kerfs: Optional[Dict[int, Any]] = None,
) -> CutPlan:
    """为一个合法排样结果生成切割段。

    参数
    ----
    parts: PolyPart 序列(1..MAX_PARTS 个)。
    placements: PolyPlacement 序列, 每个零件恰好一次(下标完整、不重复),
      描述平移放置; 放置后零件必须互不重叠且坐标非负, 否则报错。
    kerf: 默认切缝补偿宽度(非负有理数, int/Fraction)。
    kerfs: 可选的按零件下标覆盖的切缝宽度 ``{part_index: width}``;
      只有完全重合 **且切缝补偿相同** 的边才合并为一刀。

    返回 :class:`CutPlan`。输入不合法时抛 ``TypeError`` /
    ``InvalidSpecError``, 不会产生部分结果。
    """
    # --- parts ---
    if isinstance(parts, (str, bytes)):
        raise TypeError("parts: 需要 PolyPart 序列")
    try:
        parts = list(parts)
    except TypeError:
        raise TypeError("parts: 需要 PolyPart 序列") from None
    for i, p in enumerate(parts):
        if not isinstance(p, PolyPart):
            raise TypeError(
                f"parts[{i}]: 需要 PolyPart, 得到 {type(p).__name__}"
            )
    if not (1 <= len(parts) <= MAX_PARTS):
        raise InvalidSpecError(
            f"parts: 零件数 {len(parts)} 超出范围 [1, {MAX_PARTS}]"
        )

    # --- placements ---
    if isinstance(placements, (str, bytes)):
        raise TypeError("placements: 需要 PolyPlacement 序列")
    try:
        placements = list(placements)
    except TypeError:
        raise TypeError("placements: 需要 PolyPlacement 序列") from None
    for i, pl in enumerate(placements):
        if not isinstance(pl, PolyPlacement):
            raise TypeError(
                f"placements[{i}]: 需要 PolyPlacement, 得到 {type(pl).__name__}"
            )

    # --- kerf ---
    full_kerfs = _resolve_kerfs(len(parts), kerf, kerfs)

    # --- 平移量(Fraction) ---
    translations: List[Tuple[Fraction, Fraction]] = []
    for k, pl in enumerate(placements):
        tx = _parse_rational(pl.tx, f"placements[{k}].tx")
        ty = _parse_rational(pl.ty, f"placements[{k}].ty")
        translations.append((tx, ty))

    # --- 放置合法性(下标、越界、重叠) ---
    _check_poly_placements(parts, placements, translations)

    # placements 可能不按 part_index 顺序: 重排为按零件下标
    order = sorted(range(len(placements)), key=lambda k: placements[k].part_index)
    polygons = [parts[placements[k].part_index].contours() for k in order]
    trans_ordered = [translations[k] for k in order]
    kerfs_ordered = [
        full_kerfs[placements[k].part_index] for k in order
    ]

    return _cut_plan_for_parts(polygons, trans_ordered, kerfs_ordered)


# ---------------------------------------------------------------------------
# 从网格排样结果(pack 的 Placement)生成切割段
# ---------------------------------------------------------------------------


def cut_segments_for_layout(
    board: Board,
    parts: Sequence[Part],
    placements: Sequence[Placement],
    height: int,
    kerf: Any = 0,
    kerfs: Optional[Dict[int, Any]] = None,
) -> CutPlan:
    """为网格 :func:`pack` 的合法排样结果生成切割段。

    把每个放置后的格子零件(取其实际旋转/镜像方向)的边界提取为有理
    整数直边轮廓(含孔洞, 如 3x3 环), 再按 :func:`cut_segments` 的规则
    共边合并: 完全重合且切缝补偿相同的边只切一刀, 保留涉及的零件 ID;
    部分重合在交点处切分; 内外轮廓分别成环。

    切割语义允许零件沿边相接(这是共边切割的前提), 因此这里 **不** 套用
    排样的"一格 Chebyshev 边距", 只复核: 方向被允许、格在板/高度内、
    不覆禁放格、零件填充区域无正面积重叠。布局可直接取自
    ``PackResult.placements``(该结果已由排样器保证满足一格边距)。

    参数同 :func:`verify_layout`, 另加 ``kerf`` / ``kerfs``。
    """
    if not isinstance(board, Board):
        raise TypeError(f"board: 需要 Board, 得到 {type(board).__name__}")
    if isinstance(parts, (str, bytes)):
        raise TypeError("parts: 需要 Part 序列")
    try:
        parts = list(parts)
    except TypeError:
        raise TypeError("parts: 需要 Part 序列") from None
    for i, p in enumerate(parts):
        if not isinstance(p, Part):
            raise TypeError(f"parts[{i}]: 需要 Part, 得到 {type(p).__name__}")
    if isinstance(placements, (str, bytes)):
        raise TypeError("placements: 需要 Placement 序列")
    try:
        placements = list(placements)
    except TypeError:
        raise TypeError("placements: 需要 Placement 序列") from None
    for i, pl in enumerate(placements):
        if not isinstance(pl, Placement):
            raise TypeError(
                f"placements[{i}]: 需要 Placement, 得到 {type(pl).__name__}"
            )
    height = _require_int(height, "height")
    if not (1 <= height <= board.height):
        raise InvalidSpecError(
            f"height: {height} 超出范围 [1, {board.height}]"
        )

    # 每个零件: 校验并构造放置后的轮廓环 + 平移
    seen = set()
    polygons: List[Tuple[Tuple[_Pt, ...], ...]] = [None] * len(parts)  # type: ignore
    translations: List[Tuple[Fraction, Fraction]] = [(Fraction(0), Fraction(0))] * len(parts)
    for k, pl in enumerate(placements):
        i = pl.part_index
        if not isinstance(i, int) or isinstance(i, bool) or not (0 <= i < len(parts)):
            raise InvalidSpecError(f"placements[{k}].part_index: 非法下标 {i!r}")
        if i in seen:
            raise InvalidSpecError(f"placements: 零件 {i} 被重复放置")
        seen.add(i)
        part = parts[i]
        px = _require_int(pl.x, f"placements[{k}].x")
        py = _require_int(pl.y, f"placements[{k}].y")
        if pl.rotation_degrees not in _VALID_ROTATIONS:
            raise InvalidSpecError(
                f"零件 {i}: 非法旋转角 {pl.rotation_degrees}"
            )
        if not isinstance(pl.mirrored, bool):
            raise TypeError(f"零件 {i}: mirrored 需要 bool")
        if pl.rotation_degrees not in part.rotations:
            raise InvalidSpecError(
                f"零件 {i}: 旋转角 {pl.rotation_degrees} 不在允许集合内"
            )
        if pl.mirrored and not part.allow_reflection:
            raise InvalidSpecError(f"零件 {i}: 使用了镜像但该零件禁用镜像")
        shape = oriented_cells(part.cells, pl.rotation_degrees, pl.mirrored)
        abs_cells = [(x + px, y + py) for x, y in shape]
        for x, y in abs_cells:
            if not (0 <= x < board.width and 0 <= y < height):
                raise InvalidSpecError(
                    f"零件 {i}: 格子 ({x}, {y}) 超出板/高度范围"
                )
            if (x, y) in board.forbidden:
                raise InvalidSpecError(
                    f"零件 {i}: 格子 ({x}, {y}) 覆盖禁放格"
                )
        polygons[i] = _grid_contours(abs_cells)
        translations[i] = (Fraction(0), Fraction(0))

    if len(seen) != len(parts):
        missing = sorted(set(range(len(parts))) - seen)
        raise InvalidSpecError(f"placements: 缺少零件 {missing} 的放置")

    # 填充区域无正面积重叠(允许贴边)
    for a in range(len(parts)):
        for b in range(a + 1, len(parts)):
            if _regions_overlap(polygons[a], polygons[b]):
                raise InvalidSpecError(
                    f"零件 {a} 与零件 {b} 的排样区域互相重叠"
                )

    full_kerfs = _resolve_kerfs(len(parts), kerf, kerfs)
    # polygons 已按零件下标排列(绝对坐标, 平移为 0)
    return _cut_plan_for_parts(polygons, translations, full_kerfs)
