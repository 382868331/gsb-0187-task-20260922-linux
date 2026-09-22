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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "MAX_BOARD_WIDTH",
    "MAX_BOARD_HEIGHT",
    "MAX_PARTS",
    "MAX_PART_CELLS",
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
]

# ---------------------------------------------------------------------------
# 公开常量(接口边界)
# ---------------------------------------------------------------------------

MAX_BOARD_WIDTH = 20
MAX_BOARD_HEIGHT = 20
MAX_PARTS = 12
MAX_PART_CELLS = 12
DEFAULT_BUDGET = 1_000_000

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
