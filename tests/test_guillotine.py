"""整刀切割检查的测试。

期望值来源: 手工推导的小规模算例 + tests/_guillotine_ref.py 的独立
暴力参考(不调用被测核心)。
"""

import random
import unittest

from nesting2d import (
    Board,
    BlockedCut,
    Cut,
    CutBranch,
    CutLeaf,
    GuillotineStatus,
    InvalidSpecError,
    Part,
    Placement,
    guillotine_cut,
)

import _guillotine_ref


def pl(i, x, y):
    return Placement(i, x, y, 0, False)


def cuttable_case():
    """6x3: P0 方块@(0,0), P1 L@(3,0), P2 竖多米诺@(5,1)。"""
    board = Board(6, 3)
    parts = [
        Part([(0, 0), (1, 0), (0, 1), (1, 1)]),
        Part([(0, 0), (1, 0), (1, 1)]),
        Part([(0, 0), (0, 1)]),
    ]
    placements = (pl(0, 0, 0), pl(1, 3, 0), pl(2, 5, 1))
    return board, parts, placements


def windmill_case():
    """4x4 风车(5 件铺满): 四 L 角件 + 中心方块, 无第一刀。"""
    board = Board(4, 4)
    parts = [
        Part([(0, 0), (0, 1), (1, 1)]),       # 0: 左上 L @ (0,2)
        Part([(0, 1), (1, 1), (1, 0)]),       # 1: 右上 L @ (2,2)
        Part([(1, 0), (1, 1), (0, 0)]),       # 2: 右下 L @ (2,0)
        Part([(0, 0), (0, 1), (1, 0)]),       # 3: 左下 L @ (0,0)
        Part([(0, 0), (1, 0), (0, 1), (1, 1)]),  # 4: 中心方块 @ (1,1)
    ]
    placements = (
        pl(0, 0, 2), pl(1, 2, 2), pl(2, 2, 0), pl(3, 0, 0), pl(4, 1, 1)
    )
    return board, parts, placements


class TestCuttable(unittest.TestCase):
    def test_basic_cuttable(self):
        board, parts, placements = cuttable_case()
        res = guillotine_cut(board, parts, placements)
        self.assertIs(res.status, GuillotineStatus.CUTTABLE)
        self.assertEqual(len(res.cuts), 2)
        self.assertEqual(res.total_length, 6)
        self.assertEqual(res.cuts[0], Cut("v", 2, (0, 0, 6, 3)))
        self.assertEqual(res.cuts[1], Cut("v", 5, (2, 0, 6, 3)))

    def test_tree_structure(self):
        board, parts, placements = cuttable_case()
        res = guillotine_cut(board, parts, placements)
        root = res.tree
        self.assertIsInstance(root, CutBranch)
        self.assertEqual(root.cut, Cut("v", 2, (0, 0, 6, 3)))
        self.assertEqual(root.lower, CutLeaf((0, 0, 2, 3), 0))
        self.assertIsInstance(root.upper, CutBranch)
        leaves = []

        def walk(node):
            if isinstance(node, CutLeaf):
                leaves.append(node)
            else:
                walk(node.lower)
                walk(node.upper)

        walk(res.tree)
        self.assertEqual(sorted(leaf.part_index for leaf in leaves), [0, 1, 2])
        for leaf in leaves:
            self.assertIsNotNone(leaf.part_index)

    def test_single_part_no_cuts(self):
        board = Board(3, 3)
        parts = [Part([(0, 0), (1, 0)])]
        res = guillotine_cut(board, parts, [pl(0, 1, 1)])
        self.assertIs(res.status, GuillotineStatus.CUTTABLE)
        self.assertEqual(res.cuts, ())
        self.assertEqual(res.total_length, 0)
        self.assertEqual(res.tree, CutLeaf((0, 0, 3, 3), 0))

    def test_touching_parts_split_along_boundary(self):
        """不同零件相邻共享边: 沿零件边界整刀合法(不穿任何零件内部)。"""
        board = Board(2, 1)
        parts = [Part([(0, 0)]), Part([(0, 0)])]
        res = guillotine_cut(board, parts, [pl(0, 0, 0), pl(1, 1, 0)])
        self.assertIs(res.status, GuillotineStatus.CUTTABLE)
        self.assertEqual(res.cuts[0], Cut("v", 1, (0, 0, 2, 1)))

    def test_lexicographic_tie_break(self):
        """1x3 上两个单件格在列 0 和 2: 刀数与切长相同, 选坐标小的 x=1。"""
        board = Board(3, 1)
        parts = [Part([(0, 0)]), Part([(0, 0)])]
        res = guillotine_cut(board, parts, [pl(0, 0, 0), pl(1, 2, 0)])
        self.assertEqual(res.cuts[0].coord, 1)

    def test_no_zero_width_pieces(self):
        """每刀严格在子板内部: 所有叶/刀矩形均非退化。"""
        board, parts, placements = cuttable_case()
        res = guillotine_cut(board, parts, placements)

        def check(node):
            x0, y0, x1, y1 = node.rect if isinstance(node, CutLeaf) else node.cut.rect
            self.assertTrue(x0 < x1 and y0 < y1, msg=str(node))
            if isinstance(node, CutBranch):
                check(node.lower)
                check(node.upper)

        check(res.tree)


class TestNotGuillotineable(unittest.TestCase):
    def test_windmill_has_no_first_cut(self):
        board, parts, placements = windmill_case()
        res = guillotine_cut(board, parts, placements)
        self.assertIs(res.status, GuillotineStatus.NOT_GUILLOTINEABLE)
        self.assertIsNone(res.cuts)
        self.assertIsNone(res.tree)
        # 根子板 4x4 全部 6 条候选切线都必须被挡住
        self.assertEqual(len(res.blocking), 6)
        blocked = {(b.axis, b.coord): b.parts for b in res.blocking}
        self.assertEqual(blocked[("v", 1)], (0, 3))
        self.assertEqual(blocked[("v", 2)], (4,))
        self.assertEqual(blocked[("v", 3)], (1, 2))
        self.assertEqual(blocked[("h", 1)], (2, 3))
        self.assertEqual(blocked[("h", 2)], (4,))
        self.assertEqual(blocked[("h", 3)], (0, 1))
        for entry in res.blocking:
            self.assertTrue(entry.parts)
            self.assertIsInstance(entry, BlockedCut)

    def test_simple_uncuttable_then_possible_after_change(self):
        """两个对角 L 互相跨过中线 -> 不可切; 移开后同一切线释放。"""
        board = Board(3, 3)
        parts = [
            Part([(0, 0), (1, 0), (0, 1)]),
            Part([(0, 0), (1, 0), (1, 1)]),
        ]
        placements = (pl(0, 0, 1), pl(1, 1, 0))
        res = guillotine_cut(board, parts, placements)
        self.assertIs(res.status, GuillotineStatus.NOT_GUILLOTINEABLE)
        # 4x2 上左右分开: x=2 沿边界整刀, 可切
        board2 = Board(4, 2)
        placements2 = (pl(0, 0, 0), pl(1, 2, 0))
        res2 = guillotine_cut(board2, parts, placements2)
        self.assertIs(res2.status, GuillotineStatus.CUTTABLE)
        self.assertEqual(res2.cuts[0], Cut("v", 2, (0, 0, 4, 2)))

    def test_not_all_layouts_declared_cuttable(self):
        """防线: 任意无重叠排样绝不被无条件宣布可切。"""
        board, parts, placements = windmill_case()
        # 无禁放格、无重叠、铺满整板, 但结论必须是不可切
        res = guillotine_cut(board, parts, placements)
        self.assertIsNot(res.status, GuillotineStatus.CUTTABLE)


class TestBudget(unittest.TestCase):
    def test_budget_exhausted_is_distinct(self):
        board, parts, placements = cuttable_case()
        res = guillotine_cut(board, parts, placements, budget=1)
        self.assertIs(res.status, GuillotineStatus.BUDGET_EXHAUSTED)
        self.assertIsNone(res.cuts)
        self.assertIsNone(res.tree)
        self.assertIsNone(res.blocking)
        # 同例调大预算可定论为可切
        ok = guillotine_cut(board, parts, placements)
        self.assertIs(ok.status, GuillotineStatus.CUTTABLE)

    def test_small_budget_still_proves_windmill(self):
        """风车第一刀即被全部挡住: 即使预算很小也能证明不可切。"""
        board, parts, placements = windmill_case()
        res = guillotine_cut(board, parts, placements, budget=1000)
        self.assertIs(res.status, GuillotineStatus.NOT_GUILLOTINEABLE)
        self.assertEqual(res.nodes, 1)

    def test_budget_argument_validation(self):
        board, parts, placements = cuttable_case()
        with self.assertRaises(InvalidSpecError):
            guillotine_cut(board, parts, placements, budget=0)
        with self.assertRaises(TypeError):
            guillotine_cut(board, parts, placements, budget=True)


class TestErrors(unittest.TestCase):
    def test_overlap_rejected(self):
        board = Board(2, 1)
        parts = [Part([(0, 0)]), Part([(0, 0)])]
        with self.assertRaises(InvalidSpecError):
            guillotine_cut(board, parts, [pl(0, 0, 0), pl(1, 0, 0)])

    def test_out_of_board_rejected(self):
        board = Board(2, 2)
        parts = [Part([(0, 0)]), Part([(0, 0)])]
        with self.assertRaises(InvalidSpecError):
            guillotine_cut(board, parts, [pl(0, 0, 0), pl(1, 2, 0)])

    def test_forbidden_rejected(self):
        board = Board(2, 2, forbidden=[(1, 1)])
        parts = [Part([(0, 0)]), Part([(0, 0)])]
        with self.assertRaises(InvalidSpecError):
            guillotine_cut(board, parts, [pl(0, 0, 0), pl(1, 1, 1)])

    def test_count_mismatch_rejected(self):
        board = Board(3, 1)
        parts = [Part([(0, 0)]), Part([(0, 0)])]
        with self.assertRaises(InvalidSpecError):
            guillotine_cut(board, parts, [pl(0, 0, 0)])

    def test_duplicate_part_index_rejected(self):
        board = Board(3, 1)
        parts = [Part([(0, 0)]), Part([(0, 0)])]
        with self.assertRaises(InvalidSpecError):
            guillotine_cut(board, parts, [pl(0, 0, 0), pl(0, 2, 0)])

    def test_margin_violation_is_allowed(self):
        """边距是排样约束: 紧贴的合法无重叠布局仍可做整刀检查。"""
        board = Board(2, 1)
        parts = [Part([(0, 0)]), Part([(0, 0)])]
        res = guillotine_cut(board, parts, [pl(0, 0, 0), pl(1, 1, 0)])
        self.assertIs(res.status, GuillotineStatus.CUTTABLE)

    def test_bad_argument_types(self):
        board, parts, placements = cuttable_case()
        with self.assertRaises(TypeError):
            guillotine_cut("not a board", parts, placements)
        with self.assertRaises(TypeError):
            guillotine_cut(board, ["not a part"], placements)
        with self.assertRaises(TypeError):
            guillotine_cut(board, parts, [("not", "a", "placement")])


def _random_layout(rng):
    """在 <=6x6 板上随机放 1..4 个无重叠小零件(允许贴边)。"""
    w = rng.randint(2, 6)
    h = rng.randint(2, 6)
    board = Board(w, h)
    shapes = [
        ((0, 0),),
        ((0, 0), (1, 0)),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0), (0, 1)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (0, 1), (1, 1)),
    ]
    n = rng.randint(1, 4)
    parts = []
    placements = []
    occupied = set()
    attempts = 0
    while len(parts) < n and attempts < 100:
        attempts += 1
        shape = rng.choice(shapes)
        sw = max(x for x, _ in shape) + 1
        sh = max(y for _, y in shape) + 1
        x = rng.randrange(w - sw + 1)
        y = rng.randrange(h - sh + 1)
        cells = {(x + dx, y + dy) for dx, dy in shape}
        if cells & occupied:
            continue
        occupied |= cells
        parts.append(Part(shape))
        placements.append(pl(len(parts) - 1, x, y))
    return board, parts, placements


class TestAgainstIndependentReference(unittest.TestCase):
    def test_random_small_layouts(self):
        rng = random.Random(20260926)
        checked = 0
        cuttable_count = 0
        for _ in range(400):
            board, parts, placements = _random_layout(rng)
            if not parts:
                continue
            checked += 1
            res = guillotine_cut(board, parts, placements)
            part_cells = []
            for p, item in zip(parts, placements):
                shape = tuple(sorted(p.cells))
                part_cells.append(frozenset((x + item.x, y + item.y) for x, y in shape))
            ref = _guillotine_ref.best_cut(part_cells, (0, 0, board.width, board.height))
            if ref is None:
                self.assertIs(
                    res.status, GuillotineStatus.NOT_GUILLOTINEABLE,
                    msg=str(placements),
                )
            else:
                cuttable_count += 1
                self.assertIs(res.status, GuillotineStatus.CUTTABLE)
                self.assertEqual(len(res.cuts), ref[0])
                self.assertEqual(res.total_length, ref[1])
        self.assertGreater(checked, 200)
        self.assertGreater(cuttable_count, 10)
