"""约束二维材料片的无重叠排样（nesting）核心模块。

在整数网格板上排布若干连通格子零件，目标依次为：
  1. 最小化使用高度（所有零件实际占用格子的最大行号 + 1，行 0 为底部）；
  2. 同高度下最小化旋转次数（仅计旋转角非 0 的零件数，镜像不计）；
  3. 再按布局坐标字典序取最小（按零件输入顺序拼接 (x, y, code)，
     code = 旋转序号 * 2 + 镜像位）。

约束：
  * 零件格子必须落在板内且不覆盖禁放格；
  * 任意两个不同零件的占用格子之间 Chebyshev 距离 >= 2，即零件
    之间至少隔一个空格。实现上等价于：新零件的占用格不得落入
    已放零件的一格 Chebyshev 膨胀区（膨胀区本身允许相互重叠、
    允许越出板外、允许覆盖禁放格；板外不计额外空间）；
  * 每个零件仅可使用其声明允许的旋转角（0/90/180/270 度）与镜像。

实现：位集（Python int 位图）回溯 + 等形零件对称去重 + 显式搜索预算。
预算耗尽时报告当前最优已知解与已证下界，绝不把未证最优的结果标为最优。

仅使用 Python 3.14 标准库。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "NestingError",
    "Piece",
    "Board",
    "Placement",
    "NestingResult",
    "solve",
    "placed_cells",
    "render",
    "MAX_BOARD_DIM",
    "MAX_PIECES",
    "DEFAULT_BUDGET",
    "STATUS_OPTIMAL",
    "STATUS_INFEASIBLE",
    "STATUS_BUDGET_EXHAUSTED",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

MAX_BOARD_DIM = 20          # 板宽/高上限（题面接口边界）
MAX_PIECES = 12             # 零件数上限（题面接口边界）
DEFAULT_BUDGET = 200_000    # 默认搜索预算（回溯节点数，各阶段共享）

STATUS_OPTIMAL = "optimal"
STATUS_INFEASIBLE = "infeasible"
STATUS_BUDGET_EXHAUSTED = "budget_exhausted"

_ROTATIONS = (0, 90, 180, 270)


# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------

class NestingError(ValueError):
    """所有输入校验失败的统一异常。

    message 以 ``[定位路径]`` 开头，例如 ``[pieces[2].cells]``，
    使调用方可以定位到具体非法字段。继承 ValueError 便于通用捕获。
    """


def _fail(path: str, msg: str) -> None:
    raise NestingError(f"[{path}] {msg}")


# ---------------------------------------------------------------------------
# 输入校验辅助
# ---------------------------------------------------------------------------

def _check_int(value, path: str) -> int:
    """严格整数校验：拒绝 bool、float（含 NaN/Infinity）及其他类型。"""
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, f"需要 int，得到 {type(value).__name__}: {value!r}")
    return value


def _check_cell(cell, path: str) -> Tuple[int, int]:
    if not (isinstance(cell, tuple) and len(cell) == 2):
        _fail(path, f"格子坐标需为 (x, y) 二元组，得到 {cell!r}")
    x = _check_int(cell[0], f"{path}[0]")
    y = _check_int(cell[1], f"{path}[1]")
    return (x, y)


def _normalize(cells: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    """平移归一化：最小 x、最小 y 平移到 0，返回排序后的元组。"""
    min_x = min(c[0] for c in cells)
    min_y = min(c[1] for c in cells)
    return tuple(sorted((c[0] - min_x, c[1] - min_y) for c in cells))


def _is_connected(cells: Sequence[Tuple[int, int]]) -> bool:
    """4-邻接连通性检查（BFS）。"""
    cell_set = set(cells)
    seen = {cells[0]}
    stack = [cells[0]]
    while stack:
        x, y = stack.pop()
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nb in cell_set and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cell_set)


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Piece:
    """一个零件：连通的格子集合 + 允许的朝向策略。

    cells:           非空、无重复、4-连通的 (x, y) 格子序列（任意整数坐标，
                     内部按平移归一化）。
    allowed_rotations: 允许的旋转角子集，元素取自 (0, 90, 180, 270)，
                     非空、无重复。
    allow_mirror:    是否允许镜像，必须严格为 bool。
    """

    cells: Tuple[Tuple[int, int], ...]
    allowed_rotations: Tuple[int, ...] = (0, 90, 180, 270)
    allow_mirror: bool = False

    def __post_init__(self):
        path = "Piece"
        cells = tuple(self.cells)
        if not cells:
            _fail(f"{path}.cells", "零件至少包含 1 个格子")
        norm_cells = tuple(
            _check_cell(c, f"{path}.cells[{i}]") for i, c in enumerate(cells)
        )
        if len(set(norm_cells)) != len(norm_cells):
            _fail(f"{path}.cells", "存在重复格子")
        if not _is_connected(norm_cells):
            _fail(f"{path}.cells", "格子集合不 4-连通")

        rots = tuple(self.allowed_rotations)
        if not rots:
            _fail(f"{path}.allowed_rotations", "旋转角集合不能为空")
        norm_rots = []
        for i, r in enumerate(rots):
            r = _check_int(r, f"{path}.allowed_rotations[{i}]")
            if r not in _ROTATIONS:
                _fail(
                    f"{path}.allowed_rotations[{i}]",
                    f"旋转角必须为 0/90/180/270，得到 {r}",
                )
            norm_rots.append(r)
        if len(set(norm_rots)) != len(norm_rots):
            _fail(f"{path}.allowed_rotations", "存在重复旋转角")

        if not isinstance(self.allow_mirror, bool):
            _fail(
                f"{path}.allow_mirror",
                f"需要严格 bool，得到 {type(self.allow_mirror).__name__}",
            )

        object.__setattr__(self, "cells", tuple(sorted(norm_cells)))
        object.__setattr__(self, "allowed_rotations", tuple(sorted(set(norm_rots))))


@dataclass(frozen=True)
class Board:
    """矩形网格板：width x height，含禁放格集合。"""

    width: int
    height: int
    forbidden: frozenset = frozenset()

    def __post_init__(self):
        w = _check_int(self.width, "Board.width")
        h = _check_int(self.height, "Board.height")
        if not (1 <= w <= MAX_BOARD_DIM):
            _fail("Board.width", f"需在 1..{MAX_BOARD_DIM}，得到 {w}")
        if not (1 <= h <= MAX_BOARD_DIM):
            _fail("Board.height", f"需在 1..{MAX_BOARD_DIM}，得到 {h}")
        forb = set()
        for i, c in enumerate(self.forbidden):
            x, y = _check_cell(c, f"Board.forbidden[{i}]")
            if not (0 <= x < w and 0 <= y < h):
                _fail("Board.forbidden", f"禁放格 {(x, y)} 越出 {w}x{h} 板")
            forb.add((x, y))
        object.__setattr__(self, "forbidden", frozenset(forb))


@dataclass(frozen=True)
class Placement:
    """一个零件的最终放置。

    x, y:     归一化朝向后形状左下角在板上的位置（行 0 为底部）。
    rotation: 实际旋转角（0/90/180/270）。
    mirrored: 是否镜像（镜像先于旋转施加）。
    """

    piece_index: int
    x: int
    y: int
    rotation: int
    mirrored: bool


@dataclass(frozen=True)
class NestingResult:
    """求解结果。

    status:         "optimal" / "infeasible" / "budget_exhausted"。
    height:         最优（或当前最优已知）布局的使用高度；无已知布局为 None。
    rotation_count: 该布局的旋转次数；无已知布局为 None。
    placements:     按零件输入顺序的放置元组；无已知布局为 None。
    lower_bound_height: 已证明的使用高度下界。infeasible 时为板高 + 1，
                    表示“在板高内无可行排样”已被证明。
    nodes:          实际消耗的搜索节点数。
    budget:         本次求解的预算上限。
    message:        人读说明。
    """

    status: str
    height: Optional[int]
    rotation_count: Optional[int]
    placements: Optional[Tuple[Placement, ...]]
    lower_bound_height: int
    nodes: int
    budget: int
    message: str = ""

    def render(self) -> str:
        """以字符画渲染当前布局（无布局时返回状态说明）。"""
        return render(self)


# ---------------------------------------------------------------------------
# 形状变换
# ---------------------------------------------------------------------------

def _transform(cells: Sequence[Tuple[int, int]], rot: int, mirror: bool):
    """先镜像（x -> -x）再逆时针旋转 rot 度，返回归一化形状。"""
    out = []
    for x, y in cells:
        if mirror:
            x = -x
        if rot == 0:
            pass
        elif rot == 90:
            x, y = -y, x
        elif rot == 180:
            x, y = -x, -y
        elif rot == 270:
            x, y = y, -x
        out.append((x, y))
    return _normalize(out)


def _variants(piece: Piece):
    """枚举零件的全部不同归一化朝向。

    返回 [(cells, w, h, best_code)]，best_code 为产生该形状的
    (rot, mirror) 中 code = rot_index * 2 + mirror 的最小值（字典序
    平局规则：同形状取最小 code）。自身对称导致的重复朝向在此去重。
    """
    seen: Dict[Tuple[Tuple[int, int], ...], int] = {}
    for rot in piece.allowed_rotations:
        mirrors = (False, True) if piece.allow_mirror else (False,)
        for mir in mirrors:
            shape = _transform(piece.cells, rot, mir)
            code = _ROTATIONS.index(rot) * 2 + (1 if mir else 0)
            if shape not in seen or code < seen[shape]:
                seen[shape] = code
    out = []
    for shape, code in sorted(seen.items()):
        w = max(c[0] for c in shape) + 1
        h = max(c[1] for c in shape) + 1
        out.append((shape, w, h, code))
    return out


# ---------------------------------------------------------------------------
# 位图工具
# ---------------------------------------------------------------------------

def _bits_of_cells(cells, width: int) -> int:
    m = 0
    for x, y in cells:
        m |= 1 << (y * width + x)
    return m


def _make_dilate(width: int, height: int):
    """生成按行感知的 Chebyshev 一格膨胀函数（结果裁剪在板内）。

    膨胀只用于“新零件膨胀区不得覆盖已占格”的检查；越界部分直接
    丢弃（板外不计空间），允许覆盖禁放格。
    """
    board_mask = (1 << (width * height)) - 1
    # 列掩码：第 0 列 / 第 width-1 列（防止移位跨行回绕）。
    # 位索引 = y * width + x，故 <<1 使 x 增大、>>1 使 x 减小。
    col0 = 0
    col_last = 0
    for y in range(height):
        col0 |= 1 << (y * width)
        col_last |= 1 << (y * width + width - 1)
    not_col_last = board_mask & ~col_last  # 可安全 <<1（x+1 方向）的位
    not_col0 = board_mask & ~col0          # 可安全 >>1（x-1 方向）的位

    def dilate(m: int) -> int:
        up = m << width
        down = m >> width
        return (
            m
            | ((m & not_col_last) << 1)
            | ((m & not_col0) >> 1)
            | up
            | down
            | ((up & not_col_last) << 1)
            | ((up & not_col0) >> 1)
            | ((down & not_col_last) << 1)
            | ((down & not_col0) >> 1)
        ) & board_mask

    # 偶列/偶行 与 奇列/奇行掩码（2x2 块折叠用；移位只允许
    # 奇列->偶列、奇行->偶行，避免跨行/跨块回绕）
    even_col = odd_col = even_row = odd_row = 0
    row_bits = (1 << width) - 1
    for y in range(height):
        row = row_bits << (y * width)
        for x in range(width):
            if x % 2 == 0:
                even_col |= 1 << (y * width + x)
            else:
                odd_col |= 1 << (y * width + x)
        if y % 2 == 0:
            even_row |= row
        else:
            odd_row |= row

    def block_count(free: int) -> int:
        """自由格位集折叠到 2x2 对齐块后非空块的数量。

        板被 (x>>1, y>>1) 索引的 2x2 块（边缘为 1 宽/高）完全划分；
        同一块内任两格 Chebyshev 距离 <= 1，故每个块至多含一个零件
        的格子。非空块数是“可放零件数”的合法必要上界。
        """
        # 奇列并入左侧偶列代表位（仅奇列源位参与右移，防止跨行回绕）
        t = (free & even_col) | ((free & odd_col) >> 1)
        # 奇行并入下方偶行（仅奇行源位参与下移）
        t = (t & even_row) | ((t & odd_row) >> width)
        return t.bit_count()

    return dilate, block_count


# ---------------------------------------------------------------------------
# 搜索引擎
# ---------------------------------------------------------------------------

class _BudgetExceeded(Exception):
    pass


class _Engine:
    """单次求解的位集回溯引擎。"""

    def __init__(self, board: Board, pieces: Sequence[Piece], budget: int):
        self.board = board
        self.W = board.width
        self.H = board.height
        self.pieces = list(pieces)
        self.n = len(pieces)
        self.budget = budget
        self.nodes = 0
        self.forbidden_mask = _bits_of_cells(board.forbidden, self.W)
        self.dilate, self.block_count = _make_dilate(self.W, self.H)

        # 每个零件的朝向变体
        self.variants = [_variants(p) for p in self.pieces]
        self.rot_cost = [
            {code: 1 if code >= 2 else 0 for code in {v[3] for v in vs}}
            for vs in self.variants
        ]

        # 等形零件分组（形状 + 朝向策略完全一致才可互换）
        groups: Dict[tuple, List[int]] = {}
        for i, p in enumerate(self.pieces):
            key = (_normalize(p.cells), p.allowed_rotations, p.allow_mirror)
            groups.setdefault(key, []).append(i)
        self.group_of = [0] * self.n
        for gi, members in enumerate(groups.values()):
            for i in members:
                self.group_of[i] = gi

        # 放置缓存：strip 高度 h -> 每零件的候选放置列表
        self._pl_cache: Dict[int, List[List[tuple]]] = {}

    # -- 预算 ------------------------------------------------------------

    def _tick(self):
        if self.nodes >= self.budget:
            raise _BudgetExceeded
        self.nodes += 1

    # -- 候选放置 ----------------------------------------------------------

    def _placements(self, h: int) -> List[List[tuple]]:
        """strip 高度 h 下每个零件的全部合法放置。

        每个元素为 (bits, dilated, x, y, code, rot_cost)，
        按 (x, y, code) 升序（即字典序搜索顺序；key=(x,y,code)
        同时用于等形对称去重）。
        """
        if h in self._pl_cache:
            return self._pl_cache[h]
        W = self.W
        all_pl: List[List[tuple]] = []
        for pi, vs in enumerate(self.variants):
            pls = []
            for cells, vw, vh, code in vs:
                if vw > W or vh > h:
                    continue
                rc = self.rot_cost[pi][code]
                base = _bits_of_cells(cells, W)
                for dy in range(h - vh + 1):
                    row_shift = dy * W
                    for dx in range(W - vw + 1):
                        bits = base << (row_shift + dx)
                        if bits & self.forbidden_mask:
                            continue
                        pls.append((bits, self.dilate(bits), dx, dy, code, rc))
            pls.sort(key=lambda t: (t[2], t[3], t[4]))
            all_pl.append(pls)
        self._pl_cache[h] = all_pl
        return all_pl

    # -- 布局后处理 --------------------------------------------------------

    def compact(self, recs):
        """对可行布局做“下沉”压缩：保持各零件 x 与朝向不变，逐个把
        零件沿 -y 平移直到被板面/禁放格/其他零件的膨胀区挡住，
        迭代至无变化。纯几何后处理，不消耗搜索预算，只用于改善
        报告给调用方的最优已知布局高度。返回新的记录列表。
        """
        W = self.W
        cur = {i: [x, y, code] for i, x, y, code in recs}
        # 每零件 code -> (base bits, shape)
        base_of = {}
        for i, vs in enumerate(self.variants):
            for shape, vw, vh, code in vs:
                base_of[(i, code)] = (_bits_of_cells(shape, W), shape)

        def occ(i, x, y, code):
            base, _ = base_of[(i, code)]
            bits = base << (y * W + x)
            return bits, self.dilate(bits)

        changed = True
        while changed:
            changed = False
            for i in sorted(cur, key=lambda k: cur[k][1]):
                x, y, code = cur[i]
                others_dil = 0
                for j, (xj, yj, cj) in cur.items():
                    if j == i:
                        continue
                    _, d = occ(j, xj, yj, cj)
                    others_dil |= d
                base = base_of[(i, code)][0]
                new_y = y
                for ty in range(y - 1, -1, -1):
                    bits = base << (ty * W + x)
                    if bits & (self.forbidden_mask | others_dil):
                        break
                    new_y = ty
                if new_y != y:
                    cur[i][1] = new_y
                    changed = True
        return [(i, cur[i][0], cur[i][1], cur[i][2]) for i in sorted(cur)]

    # -- 回溯搜索 ----------------------------------------------------------

    def search(
        self,
        h: int,
        rot_budget: int,
        fixed: Optional[dict] = None,
        need_layout: bool = False,
    ):
        """在 strip 高度 h 内判定可行排样（完备回溯）。

        分支策略：零件身份固定，逐件枚举其全部合法放置（位置 ×
        允许朝向，按 (x, y, code) 字典序）；每节点用 MRV 选当前
        可行候选最少的“待分支组代表”，候选列表随放置增量过滤。
        等形对称去重：每组（形状与朝向策略相同）只允许其剩余成员中
        下标最小者分支，且新放置键 (x, y, code) 必须严格大于该组
        已放置的最大键。任意解按组内键升序重贴标签后即得唯一规范
        解（零件按升序下标、升序键依次落位），故去重完备；这也使
        去重约束与任意 MRV 分支顺序相容而不产生回退抖动。

        rot_budget: 允许的最大旋转次数。
        fixed:      {piece_index: (bits, dilated, x, y, code, rc)}
                    已固定前缀，先做自洽性检查。
        need_layout: True 时返回放置记录列表，否则只判定可行性。

        返回 None（不可行）、True（可行但不需要布局）或
        [(piece_index, x, y, code), ...]（可行且需要布局）。
        """
        placements = self._placements(h)
        group_of = self.group_of
        strip_mask = (1 << (self.W * h)) - 1

        clearance = 0
        rot_used = 0
        fixed_rec = []
        # 每组已放置的键（按下标顺序追加，天然升序），只需比较末位最大键
        group_keys: Dict[int, List[tuple]] = {}
        for i, rec in (fixed or {}).items():
            bits, dil, x, y, code, rc = rec
            if bits & (clearance | self.forbidden_mask):
                return None
            clearance |= dil
            rot_used += rc
            fixed_rec.append((i, x, y, code))
            group_keys.setdefault(group_of[i], []).append((x, y, code))
        # 防御性：前缀若不是下标升序给出，按键排序保证末位为最大键
        for keys in group_keys.values():
            keys.sort()
        if rot_used > rot_budget:
            return None

        remaining = [i for i in range(self.n) if i not in (fixed or {})]
        cand = {i: list(placements[i]) for i in remaining}
        assigned: Dict[int, tuple] = dict(fixed or {})

        def dfs(rem, cand_now, clearance_mask, rot, gkeys):
            self._tick()
            if not rem:
                return True
            # 2x2 块容量剪枝：每个块至多含一个零件的格子
            free = strip_mask & ~(clearance_mask | self.forbidden_mask)
            if self.block_count(free) < len(rem):
                return False
            # 每组剩余最小下标代表
            reps: Dict[int, int] = {}
            for i in rem:
                g = group_of[i]
                if g not in reps or i < reps[g]:
                    reps[g] = i
            best_i = -1
            best_list = None
            rep_lists: Dict[int, list] = {}
            # 只过滤各组代表；未轮到的组成员不过滤（子节点再过滤）
            for g, i in reps.items():
                keys = gkeys.get(g)
                max_key = keys[-1] if keys else None
                lst = []
                for rec in cand_now.get(i, placements[i]):
                    bits, dil, x, y, code, rc = rec
                    if bits & clearance_mask:
                        continue
                    if rot + rc > rot_budget:
                        continue
                    if max_key is not None and (x, y, code) <= max_key:
                        continue
                    lst.append(rec)
                if not lst:
                    return False
                rep_lists[i] = lst
                if best_list is None or len(lst) < len(best_list):
                    best_list = lst
                    best_i = i
            g = group_of[best_i]
            bucket = gkeys.setdefault(g, [])
            for rec in best_list:
                bits, dil, x, y, code, rc = rec
                bucket.append((x, y, code))
                assigned[best_i] = rec
                # 子候选图：其他组沿用其代表已过滤列表；被选组
                # 前进到下一代表，键缺失时回退全量候选（超集，安全）
                child = {i: lst for i, lst in rep_lists.items()
                         if i != best_i}
                if dfs([j for j in rem if j != best_i], child,
                       clearance_mask | dil, rot + rc, gkeys):
                    return True
                bucket.pop()
                assigned.pop(best_i, None)
            if not bucket:
                del gkeys[g]
            return False

        if dfs(remaining, cand, clearance, rot_used, group_keys):
            if not need_layout:
                return True
            out = list(fixed_rec)
            for i in remaining:
                bits, dil, x, y, code, rc = assigned[i]
                out.append((i, x, y, code))
            out.sort()
            return out
        return None


# ---------------------------------------------------------------------------
# 主求解入口
# ---------------------------------------------------------------------------

def solve(board: Board, pieces: Sequence[Piece], budget: int = DEFAULT_BUDGET) -> NestingResult:
    """求解无重叠排样。

    参数:
        board:  Board 实例。
        pieces: Piece 序列，1..MAX_PIECES 件。
        budget: 搜索预算（回溯节点数上限，严格 int，>= 1）。

    返回 NestingResult；输入非法时抛出 NestingError。
    本函数为纯函数：不修改入参，失败时不留下任何部分变更。
    """
    if not isinstance(board, Board):
        _fail("board", f"需要 Board 实例，得到 {type(board).__name__}")
    if isinstance(pieces, (str, bytes)) or not isinstance(pieces, Sequence):
        _fail("pieces", "需要 Piece 序列")
    pieces = list(pieces)
    if not (1 <= len(pieces) <= MAX_PIECES):
        _fail("pieces", f"零件数需在 1..{MAX_PIECES}，得到 {len(pieces)}")
    for i, p in enumerate(pieces):
        if not isinstance(p, Piece):
            _fail(f"pieces[{i}]", f"需要 Piece 实例，得到 {type(p).__name__}")
    budget = _check_int(budget, "budget")
    if budget < 1:
        _fail("budget", f"预算需 >= 1，得到 {budget}")

    eng = _Engine(board, pieces, budget)
    W, H, n = eng.W, eng.H, eng.n
    total_area = sum(len(p.cells) for p in pieces)

    # 初始下界：面积下界与逐零件最小包围盒高度
    lb = max(
        -(-total_area // W),
        max(min(v[2] for v in vs) for vs in eng.variants),
        1,
    )

    def result(status, height, rot_count, recs, lb_h, msg):
        placements = None
        if recs is not None:
            by_index = {r[0]: r for r in recs}
            placements = tuple(
                Placement(
                    piece_index=i,
                    x=by_index[i][1],
                    y=by_index[i][2],
                    rotation=_ROTATIONS[by_index[i][3] >> 1],
                    mirrored=bool(by_index[i][3] & 1),
                )
                for i in range(n)
            )
        return NestingResult(
            status=status,
            height=height,
            rotation_count=rot_count,
            placements=placements,
            lower_bound_height=lb_h,
            nodes=eng.nodes,
            budget=budget,
            message=msg,
        )

    # 每零件 code -> 变体实际最高占用格行号（使用高度 = max(y+top)+1）
    code_top = [
        {code: max(c[1] for c in shape) for shape, _, _, code in vs}
        for vs in eng.variants
    ]

    def exhausted_with(recs, lb_h, template):
        """预算耗尽且已持有可行布局时构造结果（最优已知 + 已证下界）。"""
        recs = eng.compact(recs)  # 几何下沉，改善报告的最优已知高度
        rot_count = sum(1 for r in recs if r[3] >= 2)
        h_known = max(y + code_top[i][code] for i, x, y, code in recs) + 1
        return result(
            STATUS_BUDGET_EXHAUSTED, h_known, rot_count, recs, lb_h,
            template.replace("{h_known}", str(h_known)),
        )

    # ---- 阶段 0：板全高可行性探测（快速获得初始最优已知布局）-------------
    try:
        top_layout = eng.search(H, rot_budget=n, need_layout=True)
    except _BudgetExceeded:
        return result(
            STATUS_BUDGET_EXHAUSTED,
            None,
            None,
            None,
            lb,
            f"预算耗尽：连板高 {H} 的可行性都未能判定；"
            f"几何下界 {lb}（高度 < {lb} 不可能）。",
        )
    if top_layout is None:
        return result(
            STATUS_INFEASIBLE,
            None,
            None,
            None,
            H + 1,
            f"已证明在板高 {H} 内不存在可行排样（所有候选高度均被证否）。",
        )

    # ---- 阶段 1：二分搜索最小可行高度 h* --------------------------------
    # 可行性关于 strip 高度单调，二分只需 ~log2(H) 次查询。
    # 不变量：高度 <= lo 已证不可行；高度 hi 已证可行（best_layout）。
    lo = lb - 1
    hi = H
    h_star = None
    best_layout = top_layout
    try:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            recs = eng.search(mid, rot_budget=n, need_layout=True)
            if recs is not None:
                hi = mid
                best_layout = recs
            else:
                lo = mid
        h_star = hi
    except _BudgetExceeded:
        proven_lb = lo + 1
        return exhausted_with(
            best_layout, proven_lb,
            f"预算耗尽：已证明高度 < {proven_lb} 不可行，并找到高度 "
            "{h_known} 的可行布局；最小高度在两者之间，当前高度为"
            "最优已知而非已证最优。",
        )

    # ---- 阶段 2：在 h* 下二分最小可行旋转次数 r* -------------------------
    # 可行性关于旋转预算单调：预算 r 可行则更大预算必可行。
    r_lo, r_hi = -1, n
    rot_layout = best_layout
    try:
        while r_hi - r_lo > 1:
            mid = (r_lo + r_hi) // 2
            recs = eng.search(h_star, rot_budget=mid, need_layout=True)
            if recs is not None:
                r_hi = mid
                rot_layout = recs
            else:
                r_lo = mid
        r_star = r_hi
    except _BudgetExceeded:
        rot_lb = r_lo + 1
        return exhausted_with(
            rot_layout, h_star,
            f"预算耗尽：最小高度 {h_star} 已证最优，但旋转次数仅证得下界 "
            f"{rot_lb}（尚未证最优）；返回的是当前最优已知布局而非已证最优。",
        )

    # ---- 阶段 3：在 (h*, r*) 下逐位确定字典序最小布局 --------------------
    fixed: Dict[int, tuple] = {}
    committed: List[tuple] = []
    try:
        for i in range(n):
            done = False
            for rec in eng._placements(h_star)[i]:
                bits, dil, x, y, code, rc = rec
                if rc > r_star:
                    continue
                trial = dict(fixed)
                trial[i] = rec
                if eng.search(h_star, rot_budget=r_star, fixed=trial) is not None:
                    fixed = trial
                    committed.append((i, x, y, code))
                    done = True
                    break
            if not done:  # 理论上不可达：阶段 2 已证可行
                raise _BudgetExceeded
    except _BudgetExceeded:
        return exhausted_with(
            rot_layout, h_star,
            f"预算耗尽：最小高度 {h_star} 与旋转次数 {r_star} 已证最优，"
            "但布局字典序尚未证最小；返回的是当前最优已知布局。",
        )

    return result(
        STATUS_OPTIMAL,
        h_star,
        r_star,
        committed,
        h_star,
        f"已证最优：高度 {h_star}，旋转次数 {r_star}，字典序最小布局。",
    )


# ---------------------------------------------------------------------------
# 输出辅助
# ---------------------------------------------------------------------------

def placed_cells(piece: Piece, placement: Placement) -> Tuple[Tuple[int, int], ...]:
    """按放置信息计算零件实际占用的板格子集合（绝对坐标）。"""
    shape = _transform(piece.cells, placement.rotation, placement.mirrored)
    return tuple(sorted((x + placement.x, y + placement.y) for x, y in shape))


def render(result: NestingResult, board: Optional[Board] = None,
           pieces: Optional[Sequence[Piece]] = None) -> str:
    """字符画渲染布局：顶行在上，底行（y=0）在下。

    需要 board 与 pieces 才能画出零件字母；缺省时仅输出状态行。
    零件标签依次为 A..L；禁放格为 '#'，空格为 '.'。
    """
    lines = [
        f"status={result.status} height={result.height} "
        f"rotations={result.rotation_count} "
        f"lower_bound={result.lower_bound_height} "
        f"nodes={result.nodes}/{result.budget}"
    ]
    if result.message:
        lines.append(result.message)
    if result.placements is None or board is None or pieces is None:
        return "\n".join(lines)
    grid = [["."] * board.width for _ in range(board.height)]
    labels = "ABCDEFGHIJKL"
    for i, pl in enumerate(result.placements):
        for x, y in placed_cells(pieces[i], pl):
            grid[y][x] = labels[i % len(labels)]
    for x, y in board.forbidden:
        grid[y][x] = "#"
    for y in range(board.height - 1, -1, -1):
        lines.append("".join(grid[y]))
    return "\n".join(lines)
