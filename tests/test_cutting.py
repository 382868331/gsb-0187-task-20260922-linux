"""nesting2d.cutting 测试套件。

期望值来源:
* 手工推导的小算例(周长/共边计数可直接手算);
* 本文件内的独立网格 oracle ``_oracle_units`` —— 不导入 nesting2d 的
  任何内部实现, 用独立的方向变换、独立泛洪(区分内外轮廓)和逐单位边
  聚合, 在小随机布局上与被测输出的单位边分解逐一比对。
"""

import random
import unittest
from fractions import Fraction

from nesting2d import (
    Board,
    CutSegment,
    InvalidSpecError,
    Part,
    Placement,
    generate_cut_segments,
    pack,
)


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


def mono():
    return Part([(0, 0)])


def dom(rotations=(0, 90)):
    return Part([(0, 0), (1, 0)], rotations=rotations)


def ring():
    """3x3 空心环, 8 格。"""
    return Part(
        [(x, y) for x in range(3) for y in range(3) if (x, y) != (1, 1)],
        rotations=(0, 90, 180, 270),
    )


def seg_units(seg: CutSegment):
    """把输出段拆回单位边: ('H', y, x) 或 ('V', x, y)。"""
    if seg.y1 == seg.y2:
        lo = min(seg.x1, seg.x2)
        hi = max(seg.x1, seg.x2)
        return [("H", seg.y1, x) for x in range(lo, hi)]
    lo = min(seg.y1, seg.y2)
    hi = max(seg.y1, seg.y2)
    return [("V", seg.x1, y) for y in range(lo, hi)]


# ---------------------------------------------------------------------------
# 独立 oracle(不使用 nesting2d 内部)
# ---------------------------------------------------------------------------


def _independent_transform(cells, rot, mirrored):
    """独立的方向实现: 先 (x,y)->(-x,y) 镜像, 再逆时针旋转, 再归一化。"""
    out = []
    for x, y in cells:
        if mirrored:
            x = -x
        for _ in range(rot // 90):
            x, y = -y, x
        out.append((x, y))
    mx = min(x for x, _ in out)
    my = min(y for _, y in out)
    return frozenset((x - mx, y - my) for x, y in out)


def _flood_components(cells):
    """相对单个零件的空格分类: 0=外部, >=1=孔洞; 独立实现。"""
    min_x = min(x for x, _ in cells) - 1
    max_x = max(x for x, _ in cells) + 1
    min_y = min(y for _, y in cells) - 1
    max_y = max(y for _, y in cells) + 1
    comp = {}

    def spread(seeds, cid):
        stack = list(seeds)
        while stack:
            x, y = stack.pop()
            if (x, y) in comp:
                continue
            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                continue
            if (x, y) in cells:
                continue
            comp[(x, y)] = cid
            stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

    spread(
        [(min_x, min_y), (max_x, min_y), (min_x, max_y), (max_x, max_y)], 0
    )
    cid = 0
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if (x, y) in cells or (x, y) in comp:
                continue
            cid += 1
            spread([(x, y)], cid)
    return comp


def _oracle_units(shapes, placed_orient, anchors, kerfs):
    """期望值: {(单位边, kerf): (parts 集合, roles 集合)}。

    shapes[i]: 零件原始格子(可迭代); placed_orient[i]=(rot, mir);
    anchors[i]=(px, py)。
    """
    placed = []
    for shape, (rot, mir), (px, py) in zip(shapes, placed_orient, anchors):
        s = _independent_transform(shape, rot, mir)
        placed.append(frozenset((x + px, y + py) for x, y in s))

    expected = {}
    for i, cells in enumerate(placed):
        comp = _flood_components(cells)
        sides = []
        for x, y in cells:
            sides += [
                ((x + 1, y), ("V", x + 1, y)),
                ((x, y + 1), ("H", y + 1, x)),
                ((x - 1, y), ("V", x, y)),
                ((x, y - 1), ("H", y, x)),
            ]
        for nb, key in sides:
            if nb in cells:
                continue  # 同零件内部, 不是轮廓
            role = "outer" if comp[nb] == 0 else "inner"
            bucket = expected.setdefault((key, kerfs[i]), (set(), set()))
            bucket[0].add(i)
            bucket[1].add(role)
    return expected


def _actual_units(plan):
    actual = {}
    for seg in plan.segments:
        for key in seg_units(seg):
            bucket = actual.setdefault((key, seg.kerf_width), (set(), set()))
            bucket[0].update(seg.part_ids)
            if "outer" in seg.contours:
                bucket[1].add("outer")
            if "inner" in seg.contours:
                bucket[1].add("inner")
    return actual


# ---------------------------------------------------------------------------
# 手算正常用例
# ---------------------------------------------------------------------------


class TestSinglePart(unittest.TestCase):
    def test_monomino_perimeter(self):
        board = Board(3, 3)
        plan = generate_cut_segments(board, [mono()], [Placement(0, 0, 0, 0, False)])
        self.assertEqual(len(plan), 4)
        self.assertEqual(plan.total_length, 4)
        for s in plan.segments:
            self.assertEqual(set(s.part_ids), {0})
            self.assertEqual(s.contours, ("outer",))
            self.assertEqual(s.length, 1)
        # 端点固定排序
        for s in plan.segments:
            self.assertLessEqual((s.x1, s.y1), (s.x2, s.y2))

    def test_domino_two_horizontal_edges_length_two(self):
        board = Board(4, 4)
        plan = generate_cut_segments(board, [dom()], [Placement(0, 1, 1, 0, False)])
        # 1x2 矩形: 2 条长 2 的水平边 + 2 条长 1 的竖直边
        self.assertEqual(len(plan), 4)
        self.assertEqual(plan.total_length, 6)
        horiz = [s for s in plan.segments if s.y1 == s.y2]
        self.assertEqual(sorted((s.x1, s.x2) for s in horiz), [(1, 3), (1, 3)])
        vert = [s for s in plan.segments if s.x1 == s.x2]
        self.assertEqual({s.length for s in vert}, {1})

    def test_ring_outer_and_inner_contours(self):
        board = Board(5, 5)
        plan = generate_cut_segments(board, [ring()], [Placement(0, 1, 1, 0, False)])
        # 外轮廓 3x3 = 12, 内孔 1x1 = 4; 8 条极大段
        self.assertEqual(len(plan), 8)
        self.assertEqual(plan.total_length, 16)
        inners = [s for s in plan.segments if s.contours == ("inner",)]
        outers = [s for s in plan.segments if s.contours == ("outer",)]
        self.assertEqual(sum(s.length for s in inners), 4)
        self.assertEqual(sum(s.length for s in outers), 12)
        # 内孔边恰好是单位正方形 [2,3]x[2,3](环锚点 (1,1))
        self.assertEqual(
            sorted((s.x1, s.y1, s.x2, s.y2) for s in inners),
            [(2, 2, 2, 3), (2, 2, 3, 2), (2, 3, 3, 3), (3, 2, 3, 3)],
        )


class TestSharedEdges(unittest.TestCase):
    def test_two_adjacent_monominoes_cut_once(self):
        board = Board(3, 3)
        placements = [Placement(0, 0, 0, 0, False), Placement(1, 1, 0, 0, False)]
        plan = generate_cut_segments(board, [mono(), mono()], placements)
        # 周长和 8, 共边 1 只切一次 => 7
        self.assertEqual(plan.total_length, 7)
        self.assertEqual(len(plan), 7)
        shared = [s for s in plan.segments if s.part_ids == frozenset({0, 1})]
        self.assertEqual(len(shared), 1)
        self.assertEqual(
            (shared[0].x1, shared[0].y1, shared[0].x2, shared[0].y2),
            (1, 0, 1, 1),
        )
        self.assertEqual(shared[0].contours, ("outer",))

    def test_two_dominoes_full_coincident_edge(self):
        # 两个横多米诺上下紧贴: 完全重合的长度-2 水平边只切一次
        board = Board(4, 4)
        placements = [Placement(0, 0, 0, 0, False), Placement(1, 0, 1, 0, False)]
        plan = generate_cut_segments(board, [dom(), dom()], placements)
        # 周长和 6+6, 共边长 2 => 10
        self.assertEqual(plan.total_length, 10)
        shared = [s for s in plan.segments if s.part_ids == frozenset({0, 1})]
        self.assertEqual(len(shared), 1)
        self.assertEqual(
            (shared[0].x1, shared[0].y1, shared[0].x2, shared[0].y2),
            (0, 1, 2, 1),
        )
        self.assertEqual(shared[0].length, 2)

    def test_partial_overlap_not_over_merged(self):
        # 横多米诺(0,0)-(2,1) 顶边 [0,2]@y=1; 单格锚 (1,1) 底边 [1,2]@y=1
        board = Board(4, 4)
        placements = [Placement(0, 0, 0, 0, False), Placement(1, 1, 1, 0, False)]
        plan = generate_cut_segments(board, [dom(), mono()], placements)
        # 周长和 6+4, 重合长度 1 => 9
        self.assertEqual(plan.total_length, 9)
        line = sorted(
            (s.x1, s.x2, tuple(sorted(s.part_ids)))
            for s in plan.segments
            if s.y1 == s.y2 == 1
        )
        self.assertEqual(line, [(0, 1, (0,)), (1, 2, (0, 1))])
        # 不存在跨越 [0,2] 且同时带两个零件的段(部分重合不得整条去重)
        for s in plan.segments:
            if s.part_ids == frozenset({0, 1}):
                self.assertEqual(s.length, 1)

    def test_chain_of_three(self):
        board = Board(5, 2)
        placements = [Placement(i, i, 0, 0, False) for i in range(3)]
        plan = generate_cut_segments(board, [mono(), mono(), mono()], placements)
        # 3 个周长 4, 两条共边各切一次 => 10
        self.assertEqual(plan.total_length, 10)
        shared = sorted(
            (s.x1, tuple(sorted(s.part_ids)))
            for s in plan.segments
            if len(s.part_ids) == 2
        )
        self.assertEqual(shared, [(1, (0, 1)), (2, (1, 2))])

    def test_corner_contact_not_merged(self):
        # 对角单格仅共用一个点: 不得合并
        board = Board(3, 3)
        placements = [Placement(0, 0, 0, 0, False), Placement(1, 1, 1, 0, False)]
        plan = generate_cut_segments(board, [mono(), mono()], placements)
        self.assertEqual(plan.total_length, 8)
        self.assertEqual(len(plan), 8)
        self.assertTrue(all(len(s.part_ids) == 1 for s in plan.segments))

    def test_t_junction_split(self):
        # 横多米诺顶边中点上方接单格(T 接): 顶边在 x=1 处断开归属
        board = Board(4, 4)
        placements = [Placement(0, 0, 0, 0, False), Placement(1, 1, 1, 0, False)]
        plan = generate_cut_segments(board, [dom((0,)), mono()], placements)
        line = sorted(
            (s.x1, s.x2, tuple(sorted(s.part_ids)))
            for s in plan.segments
            if s.y1 == s.y2 == 1
        )
        self.assertEqual(line, [(0, 1, (0,)), (1, 2, (0, 1))])

    def test_rotated_shared_edge(self):
        # 竖多米诺(0,0)(0,1) 与横多米诺(1,1)(2,1) 共竖边 x=1, y∈[1,2]
        board = Board(4, 4)
        placements = [Placement(0, 0, 0, 90, False), Placement(1, 1, 1, 0, False)]
        plan = generate_cut_segments(board, [dom(), dom()], placements)
        shared = [s for s in plan.segments if s.part_ids == frozenset({0, 1})]
        self.assertEqual(
            [(s.x1, s.y1, s.x2, s.y2) for s in shared], [(1, 1, 1, 2)]
        )

    def test_mirrored_shared_edge(self):
        ell = Part([(0, 0), (1, 0), (0, 1)], rotations=(0,))
        board = Board(4, 4)
        # 镜像后形状 (0,0)(1,0)(1,1), 锚 (2,0): 与原件共竖边 x=2
        placements = [Placement(0, 0, 0, 0, False), Placement(1, 2, 0, 0, True)]
        plan = generate_cut_segments(board, [ell, ell], placements)
        shared = [s for s in plan.segments if s.part_ids == frozenset({0, 1})]
        self.assertEqual(
            [(s.x1, s.y1, s.x2, s.y2) for s in shared], [(2, 0, 2, 1)]
        )

    def test_placements_out_of_index_order(self):
        # placements 顺序不必按零件下标
        board = Board(3, 3)
        placements = [Placement(1, 1, 0, 0, False), Placement(0, 0, 0, 0, False)]
        plan = generate_cut_segments(board, [mono(), mono()], placements)
        shared = [s for s in plan.segments if s.part_ids == frozenset({0, 1})]
        self.assertEqual(len(shared), 1)


class TestInnerOuter(unittest.TestCase):
    def test_part_inside_hole_shared_inner_outer(self):
        # 环孔内放单格: 4 条共边一侧 inner 一侧 outer, 只切一次
        board = Board(5, 5)
        placements = [Placement(0, 1, 1, 0, False), Placement(1, 2, 2, 0, False)]
        plan = generate_cut_segments(board, [ring(), mono()], placements)
        # 环周长 16 + 单格 4 - 共边 4 = 16
        self.assertEqual(plan.total_length, 16)
        shared = [s for s in plan.segments if s.part_ids == frozenset({0, 1})]
        self.assertEqual(len(shared), 4)
        for s in shared:
            self.assertEqual(s.contours, ("inner", "outer"))
            self.assertEqual(s.contour, "shared")

    def test_ring_filled_completely(self):
        # 环孔被单格填满: 仍然 4 条 inner/outer 共边
        board = Board(3, 3)
        placements = [Placement(0, 0, 0, 0, False), Placement(1, 1, 1, 0, False)]
        plan = generate_cut_segments(board, [ring(), mono()], placements)
        shared = [s for s in plan.segments if s.part_ids == frozenset({0, 1})]
        self.assertEqual(sum(s.length for s in shared), 4)
        self.assertTrue(all(s.contours == ("inner", "outer") for s in shared))


class TestKerf(unittest.TestCase):
    def test_same_kerf_merges(self):
        board = Board(3, 3)
        placements = [Placement(0, 0, 0, 0, False), Placement(1, 1, 0, 0, False)]
        plan = generate_cut_segments(
            board, [mono(), mono()], placements, kerf_width=2
        )
        shared = [s for s in plan.segments if s.part_ids == frozenset({0, 1})]
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0].kerf_width, Fraction(2))

    def test_fraction_kerf_merges(self):
        board = Board(3, 3)
        placements = [Placement(0, 0, 0, 0, False), Placement(1, 1, 0, 0, False)]
        plan = generate_cut_segments(
            board, [mono(), mono()], placements, kerf_width=Fraction(1, 2)
        )
        shared = [s for s in plan.segments if s.part_ids == frozenset({0, 1})]
        self.assertEqual(shared[0].kerf_width, Fraction(1, 2))

    def test_different_kerf_cut_twice(self):
        board = Board(3, 3)
        placements = [Placement(0, 0, 0, 0, False), Placement(1, 1, 0, 0, False)]
        plan = generate_cut_segments(
            board, [mono(), mono()], placements, kerf_width=[1, 2]
        )
        # 共边几何位置 x=1, y∈[0,1] 上出现两条段(补偿不同 => 切两次)
        twins = [
            s
            for s in plan.segments
            if (s.x1, s.y1, s.x2, s.y2) == (1, 0, 1, 1)
        ]
        self.assertEqual(sorted(s.kerf_width for s in twins), [Fraction(1), Fraction(2)])
        for s in twins:
            self.assertEqual(len(s.part_ids), 1)  # 补偿不同不能共享
        # 周长和 8 - 共边1 + 多切1 = 8... 8-1=7(同 kerf 口径), 不同 kerf 计两次 => 8
        self.assertEqual(plan.total_length, 8)

    def test_per_part_sequence_equals_scalar(self):
        board = Board(3, 3)
        placements = [Placement(0, 0, 0, 0, False), Placement(1, 1, 0, 0, False)]
        a = generate_cut_segments(board, [mono(), mono()], placements, kerf_width=0)
        b = generate_cut_segments(
            board, [mono(), mono()], placements, kerf_width=[0, 0]
        )
        self.assertEqual(a.segments, b.segments)
        self.assertEqual(a.total_length, b.total_length)


class TestPackIntegration(unittest.TestCase):
    """pack 的合法输出直接可喂入切割; Chebyshev 边距意味着没有共边。"""

    def test_pack_result_compatible(self):
        board = Board(5, 5, forbidden=[(4, 3)])
        parts = [
            Part([(0, 0), (1, 0), (2, 0), (3, 0)], rotations=(0, 90)),
            Part([(0, 0), (1, 0), (0, 1), (1, 1)], rotations=(0,)),
            Part([(0, 0), (1, 0), (0, 1), (1, 1)], rotations=(0,)),
        ]
        res = pack(board, parts)
        self.assertEqual(res.status.value, "optimal")
        plan = generate_cut_segments(board, parts, res.placements)
        # 边距 >= 1 格 => 不存在共边, 每条段只属于一个零件
        self.assertTrue(all(len(s.part_ids) == 1 for s in plan.segments))
        # 总长 = 各零件周长(4 连通格集周长 = 4n - 2*内部相邻边数)
        def perimeter(shape):
            cs = set(shape)
            return sum(
                4 - sum((x + dx, y + dy) in cs for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                for x, y in cs
            )

        expected_total = 0
        for pl in res.placements:
            shape = _independent_transform(
                parts[pl.part_index].cells, pl.rotation_degrees, pl.mirrored
            )
            expected_total += perimeter(shape)
        self.assertEqual(plan.total_length, expected_total)


class TestDeterminism(unittest.TestCase):
    def test_same_input_same_output(self):
        board = Board(6, 6)
        parts = [mono(), mono(), dom(), ring()]
        placements = [
            Placement(0, 0, 0, 0, False),
            Placement(1, 1, 0, 0, False),
            Placement(2, 0, 2, 0, False),
            Placement(3, 3, 0, 0, False),
        ]
        a = generate_cut_segments(board, parts, placements, kerf_width=[0, 1, 0, 1])
        b = generate_cut_segments(board, parts, placements, kerf_width=[0, 1, 0, 1])
        self.assertEqual(a.segments, b.segments)
        # 文档约定的确定顺序键: 水平在前, 再竖直; 支撑线坐标、区间、kerf
        def order_key(s):
            if s.y1 == s.y2:
                return (0, s.y1, s.x1, s.x2, s.kerf_width)
            return (1, s.x1, s.y1, s.y2, s.kerf_width)

        self.assertEqual(a.segments, tuple(sorted(a.segments, key=order_key)))


# ---------------------------------------------------------------------------
# 错误语义
# ---------------------------------------------------------------------------


class TestCutErrors(unittest.TestCase):
    def setUp(self):
        self.board = Board(4, 4)
        self.parts = [mono(), mono()]
        self.placements = [
            Placement(0, 0, 0, 0, False),
            Placement(1, 1, 0, 0, False),
        ]

    def call(self, **kw):
        kw.setdefault("board", self.board)
        kw.setdefault("parts", self.parts)
        kw.setdefault("placements", self.placements)
        return generate_cut_segments(**kw)

    def test_type_errors(self):
        with self.assertRaises(TypeError):
            self.call(board="not-a-board")
        with self.assertRaises(TypeError):
            self.call(parts="ab")
        with self.assertRaises(TypeError):
            self.call(parts=[object(), mono()])
        with self.assertRaises(TypeError):
            self.call(placements="ab")
        with self.assertRaises(TypeError):
            self.call(placements=[(0, 0, 0, False), self.placements[1]])
        with self.assertRaises(TypeError):
            self.call(kerf_width=0.5)
        with self.assertRaises(TypeError):
            self.call(kerf_width=True)
        with self.assertRaises(TypeError):
            self.call(
                placements=[
                    Placement(0, 0, 0, "0", False),
                    self.placements[1],
                ]
            )

    def test_domain_errors(self):
        with self.assertRaises(InvalidSpecError):
            self.call(parts=[])  # 0 个零件
        with self.assertRaises(InvalidSpecError):
            self.call(parts=[mono()] * 13)
        with self.assertRaises(InvalidSpecError):
            self.call(placements=[])  # 放置数不匹配
        with self.assertRaises(InvalidSpecError):
            self.call(
                placements=[
                    Placement(0, 0, 0, 0, False),
                    Placement(0, 1, 0, 0, False),
                ]
            )  # 重复放置零件 0
        with self.assertRaises(InvalidSpecError):
            self.call(
                parts=[mono()],
                placements=[Placement(0, 3, 0, 0, False)],
                board=Board(3, 3),
            )  # 出板
        with self.assertRaises(InvalidSpecError):
            self.call(
                board=Board(3, 3, forbidden=[(0, 0)]),
                parts=[mono()],
                placements=[Placement(0, 0, 0, 0, False)],
            )  # 覆盖禁放格
        with self.assertRaises(InvalidSpecError):
            self.call(
                placements=[
                    Placement(0, 0, 0, 0, False),
                    Placement(1, 0, 0, 0, False),
                ]
            )  # 格子重叠
        with self.assertRaises(InvalidSpecError):
            self.call(
                parts=[Part([(0, 0), (1, 0)], rotations=(0,)), mono()],
                placements=[
                    Placement(0, 0, 0, 90, False),
                    self.placements[1],
                ],
            )  # 方向不被允许
        with self.assertRaises(InvalidSpecError):
            self.call(
                parts=[Part([(0, 0)], allow_reflection=False), mono()],
                placements=[
                    Placement(0, 0, 0, 0, True),
                    self.placements[1],
                ],
            )  # 禁用镜像却镜像
        with self.assertRaises(InvalidSpecError):
            self.call(kerf_width=-1)
        with self.assertRaises(InvalidSpecError):
            self.call(kerf_width=[0, 0, 0])
        with self.assertRaises(InvalidSpecError):
            self.call(
                placements=[
                    Placement(5, 0, 0, 0, False),
                    self.placements[1],
                ]
            )  # part_index 越界

    def test_error_message_locatable(self):
        try:
            self.call(kerf_width=-1)
        except InvalidSpecError as e:
            self.assertIn("kerf_width", str(e))
        else:
            self.fail("应抛 InvalidSpecError")

    def test_no_partial_state_on_failure(self):
        # 失败是异常, 不修改入参; 修正后同一批对象仍可正常使用
        with self.assertRaises(InvalidSpecError):
            self.call(kerf_width=-1)
        plan = self.call()
        self.assertEqual(plan.total_length, 7)


# ---------------------------------------------------------------------------
# 独立 oracle 随机交叉验证
# ---------------------------------------------------------------------------


def _random_layout(rng):
    """生成允许边界相接、不重叠的随机合法布局(单位/小多格件)。"""
    w, h = rng.randint(4, 6), rng.randint(4, 6)
    n = rng.randint(1, 4)
    shapes, orients, anchors = [], [], []
    occupied = set()
    for _ in range(n):
        cells = {(0, 0)}
        target = rng.randint(1, 5)
        while len(cells) < target:
            x, y = rng.choice(sorted(cells))
            cells.add(
                rng.choice([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
            )
        mirrored = rng.random() < 0.4
        rot = rng.choice((0, 90, 180, 270))
        norm = _independent_transform(cells, rot, mirrored)
        sw = max(x for x, _ in norm) + 1
        sh = max(y for _, y in norm) + 1
        placed_ok = False
        if sw <= w and sh <= h:
            for _try in range(200):
                px = rng.randrange(w - sw + 1)
                py = rng.randrange(h - sh + 1)
                abs_cells = {(x + px, y + py) for x, y in norm}
                if not (abs_cells & occupied):
                    occupied |= abs_cells
                    shapes.append(frozenset(cells))
                    orients.append((rot, mirrored))
                    anchors.append((px, py))
                    placed_ok = True
                    break
        if not placed_ok:
            break
    return (w, h), shapes, orients, anchors


class TestOracleCrossCheck(unittest.TestCase):
    def test_random_layouts(self):
        rng = random.Random(20260924)
        kerf_choices = (0, 0, 1, Fraction(1, 2))
        trials = 0
        for _ in range(80):
            (w, h), shapes, orients, anchors = _random_layout(rng)
            if not shapes:
                continue
            trials += 1
            kerfs = [rng.choice(kerf_choices) for _ in shapes]
            board = Board(w, h)
            parts = [
                Part(s, rotations=(0, 90, 180, 270), allow_reflection=True)
                for s in shapes
            ]
            placements = [
                Placement(i, px, py, rot, mir)
                for i, ((rot, mir), (px, py)) in enumerate(zip(orients, anchors))
            ]
            plan = generate_cut_segments(board, parts, placements, kerf_width=kerfs)
            expected = _oracle_units(shapes, orients, anchors, kerfs)
            actual = _actual_units(plan)
            self.assertEqual(actual, expected, f"布局: {shapes} {orients} {anchors}")
            # 总长 = 不同 (单位边, kerf) 组合数
            self.assertEqual(plan.total_length, len(expected))
        self.assertGreaterEqual(trials, 50)

    def test_ring_cases_against_oracle(self):
        shapes = [
            frozenset((x, y) for x in range(3) for y in range(3) if (x, y) != (1, 1)),
            frozenset({(0, 0)}),
        ]
        board = Board(5, 5)
        parts = [
            Part(shapes[0], rotations=(0, 90, 180, 270)),
            Part(shapes[1]),
        ]
        orients = [(0, False), (0, False)]
        anchors = [(1, 1), (2, 2)]  # 单格填入环孔
        kerfs = [Fraction(1, 3), 2]
        placements = [Placement(0, 1, 1, 0, False), Placement(1, 2, 2, 0, False)]
        plan = generate_cut_segments(board, parts, placements, kerf_width=kerfs)
        expected = _oracle_units(shapes, orients, anchors, kerfs)
        self.assertEqual(_actual_units(plan), expected)
        self.assertEqual(plan.total_length, len(expected))
        # 共边因 kerf 不同切两次, 总长度 = 16(同 kerf 口径) + 4(补偿不同多切)
        self.assertEqual(plan.total_length, 20)


if __name__ == "__main__":
    unittest.main()
