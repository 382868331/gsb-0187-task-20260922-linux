"""整刀(guillotine)可切性检查与切割树生成的测试。

期望值来源: 手工推导的小算例 + tests/_reference.py 的独立暴力递归参考
(不用记忆化, 不调用被测核心)。
"""

import random
import unittest

from nesting2d import (
    Board,
    Cut,
    GuillotineStatus,
    InvalidSpecError,
    Part,
    Placement,
    guillotine_cuts,
)

import _reference


def rect_part(w, h):
    return Part([(x, y) for x in range(w) for y in range(h)], rotations=(0,))


def layout_from_rects(rects):
    """rects: [(w, h, x, y), ...] -> (parts, placements)。"""
    parts = [rect_part(w, h) for w, h, _x, _y in rects]
    placements = tuple(
        Placement(i, x, y, 0, False) for i, (_w, _h, x, y) in enumerate(rects)
    )
    return parts, placements


def windmill():
    """风车形布局: 四个长条互相锁住, 不存在第一刀。

    9x9 板:
      A: 横条 x=0..5,  y=8     (挡住竖刀 c=1..5)
      B: 竖条 x=8,     y=3..8  (挡住横刀 c=4..8)
      C: 横条 x=3..8,  y=0     (挡住竖刀 c=4..8)
      D: 竖条 x=0,     y=0..6  (挡住横刀 c=1..6)
    """
    board = Board(9, 9)
    parts = [
        Part([(x, 0) for x in range(6)], rotations=(0,)),       # A
        Part([(0, y) for y in range(6)], rotations=(0,)),       # B
        Part([(x, 0) for x in range(6)], rotations=(0,)),       # C
        Part([(0, y) for y in range(7)], rotations=(0,)),       # D
    ]
    placements = (
        Placement(0, 0, 8, 0, False),
        Placement(1, 8, 3, 0, False),
        Placement(2, 3, 0, 0, False),
        Placement(3, 0, 0, 0, False),
    )
    return board, parts, placements


def first_cut_blockers(board, parts, placements, height):
    """对每个候选第一刀, 返回挡住它的零件下标列表(独立计算)。"""
    from nesting2d import oriented_cells

    boxes = []
    for pl in placements:
        shape = oriented_cells(
            parts[pl.part_index].cells, pl.rotation_degrees, pl.mirrored
        )
        xs = [x + pl.x for x, _ in shape]
        ys = [y + pl.y for _, y in shape]
        boxes.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1))
    blocked = {}
    for c in range(1, board.width):
        hit = [i for i, b in enumerate(boxes) if b[0] < c < b[2]]
        blocked[("v", c)] = hit
    for c in range(1, height):
        hit = [i for i, b in enumerate(boxes) if b[1] < c < b[3]]
        blocked[("h", c)] = hit
    return blocked


class TestGuillotineValidation(unittest.TestCase):
    def test_type_errors(self):
        board = Board(4, 4)
        parts, placements = layout_from_rects([(1, 1, 0, 0)])
        with self.assertRaises(TypeError):
            guillotine_cuts("not a board", parts, placements)
        with self.assertRaises(TypeError):
            guillotine_cuts(board, "parts", placements)
        with self.assertRaises(TypeError):
            guillotine_cuts(board, parts, "placements")
        with self.assertRaises(TypeError):
            guillotine_cuts(board, parts, placements, budget=True)
        with self.assertRaises(TypeError):
            guillotine_cuts(board, parts, placements, budget=1.5)

    def test_domain_errors(self):
        board = Board(4, 4)
        parts, placements = layout_from_rects([(1, 1, 0, 0)])
        with self.assertRaises(InvalidSpecError):
            guillotine_cuts(board, parts, placements, budget=0)
        with self.assertRaises(InvalidSpecError):
            guillotine_cuts(board, parts, placements, height=9)

    def test_invalid_layout_rejected(self):
        board = Board(4, 4)
        parts = [rect_part(2, 2), rect_part(2, 2)]
        # 两个零件重叠: 布局非法, 不得静默继续
        placements = (
            Placement(0, 0, 0, 0, False),
            Placement(1, 0, 0, 0, False),
        )
        with self.assertRaises(InvalidSpecError):
            guillotine_cuts(board, parts, placements)


class TestCuttable(unittest.TestCase):
    def test_two_squares_side_by_side(self):
        # 两个 2x2 左右排列(间隔 1 格): 一刀竖切即可
        board = Board(6, 2)
        parts, placements = layout_from_rects([(2, 2, 0, 0), (2, 2, 4, 0)])
        res = guillotine_cuts(board, parts, placements, height=2)
        self.assertIs(res.status, GuillotineStatus.CUTTABLE)
        self.assertEqual(res.cut_count, 1)
        self.assertEqual(res.total_cut_length, 2)
        self.assertEqual(len(res.cuts), 1)
        cut = res.cuts[0]
        self.assertEqual(cut.axis, "v")
        self.assertEqual(cut.position, 2)  # 坐标顺序最小: 贴左零件右边界
        self.assertEqual(cut.rect, (0, 0, 6, 2))
        self.assertEqual(cut.length, 2)

    def test_tree_structure_and_leaves(self):
        # 2x2 网格摆放四个 1x1(间隔 1 格), 板 5x5
        board = Board(5, 5)
        parts, placements = layout_from_rects(
            [(1, 1, 0, 0), (1, 1, 4, 0), (1, 1, 0, 4), (1, 1, 4, 4)]
        )
        res = guillotine_cuts(board, parts, placements, height=5)
        self.assertIs(res.status, GuillotineStatus.CUTTABLE)
        self.assertEqual(res.cut_count, 3)
        # 树: 每个内部节点两个孩子, 叶子是零件或废料
        leaves = []

        def walk(node):
            if node.is_leaf:
                leaves.append(node)
            else:
                self.assertEqual(len(node.children), 2)
                for ch in node.children:
                    walk(ch)

        walk(res.tree)
        part_leaves = [n for n in leaves if n.part_index is not None]
        self.assertEqual(sorted(n.part_index for n in part_leaves), [0, 1, 2, 3])
        # 零宽废料不单独成片: 所有叶子矩形面积 >= 1
        for n in leaves:
            x0, y0, x1, y1 = n.rect
            self.assertGreater(x1 - x0, 0)
            self.assertGreater(y1 - y0, 0)
        # 每刀作用的子板范围与树节点一致, 且贯穿子板
        self.assertEqual(res.cuts[0].rect, res.tree.rect)

    def test_cut_respects_subboard(self):
        # 子板外的零件不得阻挡子板内的刀:
        # 上下两个区域, 下部一个宽零件, 上部两个零件需竖刀分开
        board = Board(7, 5)
        parts, placements = layout_from_rects(
            [
                (5, 1, 0, 0),  # 底部长条 x=0..4
                (1, 1, 0, 3),  # 上部左
                (1, 1, 3, 3),  # 上部右
            ]
        )
        res = guillotine_cuts(board, parts, placements, height=5)
        self.assertIs(res.status, GuillotineStatus.CUTTABLE)
        self.assertEqual(res.cut_count, 2)
        # 第一刀横切分离底部长条, 第二刀在上部子板内竖切;
        # 刀位 y=3 使上部子板最矮(总切长 7+2=9 最短)
        first = res.cuts[0]
        self.assertEqual(first.axis, "h")
        self.assertEqual(first.position, 3)
        second = res.cuts[1]
        self.assertEqual(second.axis, "v")
        self.assertEqual(second.rect, (0, 3, 7, 5))
        self.assertEqual(second.position, 1)
        self.assertEqual(res.total_cut_length, 9)

    def test_min_length_tiebreak(self):
        # 三个 1x1 呈 L 形: (0,0), (4,0), (0,4)。最少 2 刀;
        # 最短总切长 6 有两个最优(先横 y=1 或先竖 x=1),
        # 按坐标顺序选先横刀("h" < "v")
        board = Board(5, 5)
        parts, placements = layout_from_rects(
            [(1, 1, 0, 0), (1, 1, 4, 0), (1, 1, 0, 4)]
        )
        res = guillotine_cuts(board, parts, placements, height=5)
        self.assertIs(res.status, GuillotineStatus.CUTTABLE)
        self.assertEqual(res.cut_count, 2)
        self.assertEqual(res.total_cut_length, 6)
        self.assertEqual(res.cuts[0].axis, "h")
        self.assertEqual(res.cuts[0].position, 1)


class TestWindmill(unittest.TestCase):
    def test_not_cuttable_with_witness(self):
        board, parts, placements = windmill()
        res = guillotine_cuts(board, parts, placements, height=9)
        self.assertIs(res.status, GuillotineStatus.NOT_CUTTABLE)
        self.assertIsNone(res.cuts)
        self.assertIsNone(res.tree)
        # 阻挡见证: 每一条候选第一刀都至少被一个零件挡住
        blocked = first_cut_blockers(board, parts, placements, 9)
        self.assertTrue(blocked)
        for line, hit in blocked.items():
            self.assertTrue(hit, f"候选第一刀 {line} 未被任何零件阻挡")

    def test_windmill_layout_itself_valid(self):
        from nesting2d import verify_layout

        board, parts, placements = windmill()
        self.assertEqual(verify_layout(board, parts, placements, 9), [])


class TestBudget(unittest.TestCase):
    def test_budget_exhausted_distinct_from_not_cuttable(self):
        board = Board(5, 5)
        parts, placements = layout_from_rects(
            [(1, 1, 0, 0), (1, 1, 4, 0), (1, 1, 0, 4), (1, 1, 4, 4)]
        )
        res = guillotine_cuts(board, parts, placements, height=5, budget=1)
        self.assertIs(res.status, GuillotineStatus.BUDGET_EXHAUSTED)
        self.assertIsNone(res.cuts)
        full = guillotine_cuts(board, parts, placements, height=5)
        self.assertIs(full.status, GuillotineStatus.CUTTABLE)


class TestReferenceCrossCheck(unittest.TestCase):
    """与独立暴力参考(无记忆化递归)对照最少刀数与最短总切长。"""

    def test_random_rect_layouts(self):
        rng = random.Random(20260926)
        checked = 0
        trials = 0
        while checked < 30 and trials < 4000:
            trials += 1
            w = rng.randint(3, 8)
            h = rng.randint(3, 8)
            n = rng.randint(1, 4)
            rects = []
            ok = True
            for _ in range(n):
                pw = rng.randint(1, min(3, w))
                ph = rng.randint(1, min(3, h))
                if pw * ph > 12:
                    ok = False
                    break
                x = rng.randint(0, w - pw)
                y = rng.randint(0, h - ph)
                nb = (x, y, x + pw, y + ph)
                for (_pw, _ph, ex, ey) in rects:
                    eb = (ex, ey, ex + _pw, ey + _ph)
                    # Chebyshev 一格边距
                    if not (
                        nb[2] + 1 <= eb[0]
                        or eb[2] + 1 <= nb[0]
                        or nb[3] + 1 <= eb[1]
                        or eb[3] + 1 <= nb[1]
                    ):
                        ok = False
                        break
                if not ok:
                    break
                rects.append((pw, ph, x, y))
            if not ok or not rects:
                continue
            board = Board(w, h)
            parts, placements = layout_from_rects(rects)
            res = guillotine_cuts(board, parts, placements, height=h)
            bboxes = [(x, y, x + pw, y + ph) for pw, ph, x, y in rects]
            ref = _reference.guillotine_min_cuts(bboxes, w, h)
            if ref is None:
                self.assertIs(res.status, GuillotineStatus.NOT_CUTTABLE)
            else:
                self.assertIs(res.status, GuillotineStatus.CUTTABLE)
                self.assertEqual((res.cut_count, res.total_cut_length), ref)
            checked += 1
        self.assertGreater(checked, 10)


if __name__ == "__main__":
    unittest.main()
