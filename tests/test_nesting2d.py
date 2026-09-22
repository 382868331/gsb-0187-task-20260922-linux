"""nesting2d 测试套件。

期望值来源: 手工推导的小规模算例 + tests/_reference.py 的独立穷举参考
(不调用被测核心)。verify_layout 往返校验仅作补充。
"""

import random
import unittest

from nesting2d import (
    Board,
    InvalidSpecError,
    Part,
    Status,
    pack,
    render_layout,
    verify_layout,
)

import _reference


def monomino():
    return Part([(0, 0)])


def domino(rotations=(0, 90)):
    return Part([(0, 0), (1, 0)], rotations=rotations)


def ring():
    """3x3 空心环(含孔零件), 8 格。"""
    cells = [(x, y) for x in range(3) for y in range(3) if (x, y) != (1, 1)]
    return Part(cells, rotations=(0,))


class TestValidation(unittest.TestCase):
    """错误语义: 类型错误 -> TypeError; 取值域违规 -> InvalidSpecError。"""

    def test_board_type_errors(self):
        with self.assertRaises(TypeError):
            Board(2.0, 3)
        with self.assertRaises(TypeError):
            Board(float("nan"), 3)
        with self.assertRaises(TypeError):
            Board(float("inf"), 3)
        with self.assertRaises(TypeError):
            Board(True, 3)  # bool 不是合法整数
        with self.assertRaises(TypeError):
            Board("3", 3)

    def test_board_domain_errors(self):
        with self.assertRaises(InvalidSpecError):
            Board(0, 3)
        with self.assertRaises(InvalidSpecError):
            Board(21, 3)  # 超过 20x20 上限
        with self.assertRaises(InvalidSpecError):
            Board(3, 21)
        with self.assertRaises(InvalidSpecError):
            Board(3, 3, forbidden=[(5, 0)])  # 禁放格越界
        with self.assertRaises(InvalidSpecError):
            Board(3, 3, forbidden=[(1, 1), (1, 1)])  # 重复禁放格
        with self.assertRaises(TypeError):
            Board(3, 3, forbidden=[(1.5, 1)])
        with self.assertRaises(TypeError):
            Board(3, 3, forbidden=["ab"])

    def test_part_errors(self):
        with self.assertRaises(InvalidSpecError):
            Part([])  # 空零件
        with self.assertRaises(InvalidSpecError):
            Part([(0, 0), (0, 0)])  # 重复格
        with self.assertRaises(InvalidSpecError):
            Part([(0, 0), (2, 2)])  # 不连通
        with self.assertRaises(InvalidSpecError):
            Part([(0, 0), (1, 1)])  # 仅角相邻也不算连通
        with self.assertRaises(InvalidSpecError):
            Part([(i, 0) for i in range(13)])  # 超过 12 格
        with self.assertRaises(InvalidSpecError):
            Part([(0, 0)], rotations=[])  # 空旋转集
        with self.assertRaises(InvalidSpecError):
            Part([(0, 0)], rotations=[45])
        with self.assertRaises(TypeError):
            Part([(0, 0)], rotations=[True])  # bool 拒绝
        with self.assertRaises(TypeError):
            Part([(0, 0)], rotations=[90.0])
        with self.assertRaises(TypeError):
            Part([(0, 0)], allow_reflection=1)
        with self.assertRaises(TypeError):
            Part([(float("nan"), 0)])

    def test_pack_argument_errors(self):
        board = Board(3, 3)
        with self.assertRaises(InvalidSpecError):
            pack(board, [])  # 0 个零件
        with self.assertRaises(InvalidSpecError):
            pack(board, [monomino()] * 13)  # 超过 12 件
        with self.assertRaises(TypeError):
            pack(board, "not-a-sequence")
        with self.assertRaises(TypeError):
            pack(board, [object()])
        with self.assertRaises(TypeError):
            pack("not-a-board", [monomino()])
        with self.assertRaises(InvalidSpecError):
            pack(board, [monomino()], budget=0)
        with self.assertRaises(TypeError):
            pack(board, [monomino()], budget=True)
        with self.assertRaises(TypeError):
            pack(board, [monomino()], budget=2.5)
        with self.assertRaises(TypeError):
            pack(board, [monomino()], budget=float("nan"))
        with self.assertRaises(TypeError):
            pack(board, [monomino()], allow_reflection=0)

    def test_error_message_locatable(self):
        try:
            Board(3, 3, forbidden=[(9, 0)])
        except InvalidSpecError as e:
            self.assertIn("forbidden", str(e))
        else:
            self.fail("应抛出 InvalidSpecError")


class TestHolePart(unittest.TestCase):
    """含孔零件(3x3 环)。"""

    def test_two_rings_side_by_side(self):
        board = Board(7, 6)
        parts = [ring(), ring()]
        res = pack(board, parts)
        self.assertEqual(res.status, Status.OPTIMAL)
        # 两个 3x3 环 + 一格边距: 横向 3+1+3=7 恰好放下, 高度 3
        self.assertEqual(res.height, 3)
        self.assertEqual(res.rotation_count, 0)
        self.assertEqual(res.lower_bound, 3)
        self.assertEqual(verify_layout(board, parts, res.placements, res.height), [])
        # 与独立参考一致
        ref = _reference.solve(board, parts)
        self.assertIsNotNone(ref)
        self.assertEqual(ref[0], res.height)

    def test_forbidden_inside_hole(self):
        # 禁放格位于环的孔内: 不阻挡放置
        board = Board(6, 7, forbidden=[(1, 1)])
        parts = [ring(), ring()]
        res = pack(board, parts)
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.height, 7)  # 竖向叠放 3+1+3
        self.assertEqual(verify_layout(board, parts, res.placements, res.height), [])
        ref = _reference.solve(board, parts)
        self.assertEqual(ref[0], 7)


class TestDuplicateParts(unittest.TestCase):
    """重复形: 3 个相同多米诺, 对称破缺不应漏解或报错。"""

    def test_three_dominoes(self):
        board = Board(5, 5)
        parts = [domino(), domino(), domino()]
        res = pack(board, parts)
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.height, 2)
        # 高度 2 下只能全部竖放: 旋转次数 3
        self.assertEqual(res.rotation_count, 3)
        self.assertEqual(
            res.layout,
            {0: (0, 0, 90, False), 1: (2, 0, 90, False), 2: (4, 0, 90, False)},
        )
        self.assertEqual(verify_layout(board, parts, res.placements, res.height), [])
        ref = _reference.solve(board, parts)
        self.assertEqual((ref[0], ref[1]), (2, 3))


class TestRotationOnlyFit(unittest.TestCase):
    """仅旋转后可放: 1x4 长条在 2x4 板上。"""

    def test_unrotated_infeasible(self):
        board = Board(2, 4)
        bar = Part([(0, 0), (1, 0), (2, 0), (3, 0)], rotations=(0,))
        res = pack(board, [bar])
        self.assertEqual(res.status, Status.INFEASIBLE)
        self.assertIsNone(res.height)
        self.assertEqual(res.lower_bound, 5)  # board.height + 1

    def test_rotated_fits(self):
        board = Board(2, 4)
        bar = Part([(0, 0), (1, 0), (2, 0), (3, 0)], rotations=(0, 90))
        res = pack(board, [bar])
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.height, 4)
        self.assertEqual(res.rotation_count, 1)
        self.assertEqual(res.layout, {0: (0, 0, 90, False)})

    def test_symmetric_shape_without_rotation_zero(self):
        # 单格形状完全对称, 但允许旋转集不含 0 度: 放置必须计 1 次旋转,
        # 且不得与允许 0 度的同形零件错误归组。
        board = Board(4, 5, forbidden=[(1, 1)])
        p0 = Part([(0, 0)], rotations=(0, 90, 180, 270))
        p1 = Part([(0, 0)], rotations=(90, 180, 270), allow_reflection=False)
        res = pack(board, [p0, p1])
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.height, 1)
        self.assertEqual(res.rotation_count, 1)
        self.assertEqual(res.layout[1][2], 90)  # 零件1 报告旋转 90 度
        self.assertEqual(res.layout[0][2], 0)


class TestInfeasible(unittest.TestCase):
    """不可行: 3x3 板放 5 个单格(边距下最多 4 个)。"""

    def test_five_monominoes(self):
        board = Board(3, 3)
        parts = [monomino()] * 5
        res = pack(board, parts)
        self.assertEqual(res.status, Status.INFEASIBLE)
        self.assertIsNone(res.placements)
        self.assertIsNone(_reference.solve(board, parts))

    def test_four_monominoes_fit(self):
        board = Board(3, 3)
        parts = [monomino()] * 4
        res = pack(board, parts)
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.height, 3)
        self.assertEqual(verify_layout(board, parts, res.placements, res.height), [])


class TestTieBreaks(unittest.TestCase):
    def test_rotation_count_minimized(self):
        # 两个多米诺在 4x2 板上: 高度 2 的最少旋转方案是 1 次
        # (一个横放一个竖放), 而非朴素的全竖放 2 次。
        board = Board(4, 2)
        parts = [domino(), domino()]
        res = pack(board, parts)
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.height, 2)
        self.assertEqual(res.rotation_count, 1)
        self.assertEqual(
            res.layout, {0: (0, 0, 0, False), 1: (3, 0, 90, False)}
        )
        ref = _reference.solve(board, parts)
        self.assertEqual((ref[0], ref[1]), (2, 1))

    def test_lexicographic_layout(self):
        # 两个单格: 高度 1 时字典序最小布局为 (0,0) 与 (2,0)
        board = Board(4, 4)
        parts = [monomino(), monomino()]
        res = pack(board, parts)
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.height, 1)
        self.assertEqual(res.layout, {0: (0, 0, 0, False), 1: (2, 0, 0, False)})

    def test_margin_not_required_outside_board(self):
        # 板外不计边距: 两个单格可贴板边放置于同一行
        board = Board(3, 3)
        parts = [monomino(), monomino()]
        res = pack(board, parts)
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.height, 1)
        self.assertEqual(res.layout, {0: (0, 0, 0, False), 1: (2, 0, 0, False)})


class TestBudgetExhausted(unittest.TestCase):
    """预算耗尽必须与证明无解/最优严格区分。"""

    def test_tiny_budget_reports_best_known(self):
        board = Board(5, 5)
        parts = [domino(), domino(), domino()]
        res = pack(board, parts, budget=1)
        self.assertEqual(res.status, Status.BUDGET_EXHAUSTED)
        self.assertIsNone(res.height)  # 预算太小, 尚无可行解
        self.assertIsNone(res.placements)
        self.assertGreaterEqual(res.lower_bound, 1)
        self.assertLessEqual(res.nodes, res.budget + 1)
        # 同一实例给足预算可证最优: 耗尽并非无解
        full = pack(board, parts)
        self.assertEqual(full.status, Status.OPTIMAL)
        self.assertEqual(full.height, 2)
        self.assertGreaterEqual(res.lower_bound, 1)
        self.assertLessEqual(res.lower_bound, full.height)

    def test_exhausted_is_not_infeasible(self):
        board = Board(3, 3)
        parts = [monomino()] * 5
        res = pack(board, parts, budget=1)
        # 预算耗尽: 即使实例实际不可行, 也不能报告 INFEASIBLE
        self.assertEqual(res.status, Status.BUDGET_EXHAUSTED)


class TestReferenceCrossCheck(unittest.TestCase):
    """与独立穷举参考交叉验证(小板)。"""

    def check(self, board, parts, allow_reflection=True):
        res = pack(board, parts, allow_reflection=allow_reflection)
        ref = _reference.solve(board, parts, allow_reflection=allow_reflection)
        if ref is None:
            self.assertEqual(res.status, Status.INFEASIBLE)
            return
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.height, ref[0])
        self.assertEqual(res.rotation_count, ref[1])
        # 布局字典序完全一致(同一目标定义下的唯一最小元)
        got = tuple(res.layout[i] for i in sorted(res.layout))
        self.assertEqual(got, ref[3])
        self.assertEqual(
            verify_layout(board, parts, res.placements, res.height), []
        )

    def test_fixed_cases(self):
        # demo 同款算例
        board = Board(5, 5, forbidden=[(4, 3)])
        parts = [
            Part([(0, 0), (1, 0), (2, 0), (3, 0)], rotations=(0, 90)),
            Part([(0, 0), (1, 0), (0, 1), (1, 1)], rotations=(0,)),
            Part([(0, 0), (1, 0), (0, 1), (1, 1)], rotations=(0,)),
        ]
        self.check(board, parts)
        # 镜像开关算例: L 形零件
        board2 = Board(4, 4)
        ell = Part([(0, 0), (1, 0), (0, 1)], rotations=(0, 90, 180, 270))
        self.check(board2, [ell, ell], allow_reflection=True)
        self.check(board2, [ell, ell], allow_reflection=False)

    def test_random_small_instances(self):
        rng = random.Random(20260922)

        def random_shape():
            cells = {(0, 0)}
            target = rng.randint(1, 5)
            while len(cells) < target:
                x, y = rng.choice(sorted(cells))
                nb = rng.choice([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
                cells.add(nb)
            return sorted(cells)

        for trial in range(40):
            w = rng.randint(3, 5)
            h = rng.randint(3, 5)
            n = rng.randint(1, 3)
            parts = []
            for _ in range(n):
                rots = tuple(
                    r for r in (0, 90, 180, 270) if rng.random() < 0.7
                ) or (0,)
                parts.append(
                    Part(
                        random_shape(),
                        rotations=rots,
                        allow_reflection=rng.random() < 0.5,
                    )
                )
            forbidden = [
                (x, y)
                for x in range(w)
                for y in range(h)
                if rng.random() < 0.1
            ]
            board = Board(w, h, forbidden=forbidden)
            with self.subTest(trial=trial, w=w, h=h, n=n):
                self.check(board, parts, allow_reflection=rng.random() < 0.5)


class TestDilateBits(unittest.TestCase):
    """白盒: 位集 Chebyshev 膨胀与朴素集合格子膨胀逐位一致。"""

    def test_dilate_matches_naive(self):
        import nesting2d

        rng = random.Random(4)
        for w in (1, 2, 3, 5, 20):
            for _ in range(30):
                cells = set()
                for _ in range(rng.randint(1, 12)):
                    cells.add((rng.randrange(w), rng.randrange(20)))
                mask = 0
                for x, y in cells:
                    mask |= 1 << (y * w + x)
                got = nesting2d._dilate(mask, w)
                want = 0
                for x, y in cells:
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < w and 0 <= ny:
                                want |= 1 << (ny * w + nx)
                self.assertEqual(got, want, f"w={w} cells={cells}")


class TestRenderAndVerify(unittest.TestCase):
    def test_render_and_verify_roundtrip(self):
        board = Board(5, 5, forbidden=[(4, 3)])
        parts = [
            Part([(0, 0), (1, 0), (2, 0), (3, 0)], rotations=(0, 90)),
            Part([(0, 0), (1, 0), (0, 1), (1, 1)], rotations=(0,)),
            Part([(0, 0), (1, 0), (0, 1), (1, 1)], rotations=(0,)),
        ]
        res = pack(board, parts)
        self.assertEqual(res.status, Status.OPTIMAL)
        art = render_layout(board, parts, res.placements, res.height)
        lines = art.split("\n")
        self.assertEqual(len(lines), res.height)
        self.assertTrue(all(len(line) == 5 for line in lines))
        self.assertIn("#", art)  # 禁放格
        for label in "012":
            self.assertIn(label, art)

    def test_verify_detects_margin_violation(self):
        board = Board(4, 4)
        parts = [monomino(), monomino()]
        from nesting2d import Placement

        bad = [Placement(0, 0, 0, 0, False), Placement(1, 1, 0, 0, False)]
        problems = verify_layout(board, parts, bad, 4)
        self.assertTrue(any("边距" in p for p in problems))

    def test_verify_detects_forbidden_overlap(self):
        board = Board(4, 4, forbidden=[(1, 1)])
        parts = [monomino()]
        from nesting2d import Placement

        problems = verify_layout(board, parts, [Placement(0, 1, 1, 0, False)], 4)
        self.assertTrue(any("禁放" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
