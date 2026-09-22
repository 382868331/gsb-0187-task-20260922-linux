"""核心求解器功能测试：边距规则、朝向策略、平局规则、状态语义。"""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import nesting
from helpers import validate_layout
from nesting import (
    Board,
    Piece,
    STATUS_BUDGET_EXHAUSTED,
    STATUS_INFEASIBLE,
    STATUS_OPTIMAL,
    solve,
)

MONO = ((0, 0),)
DOMINO = ((0, 0), (1, 0))
L3 = ((0, 0), (1, 0), (0, 1))
SQUARE2 = ((0, 0), (1, 0), (0, 1), (1, 1))
# 3x3 去心环（含孔零件，8 格，4-连通）
RING = tuple((x, y) for x in range(3) for y in range(3) if (x, y) != (1, 1))


def assert_valid(testcase, board, pieces, result):
    testcase.assertIsNotNone(result.placements)
    err = validate_layout(board, pieces, result.placements, result.height)
    testcase.assertIsNone(err, err)


class TestMarginRules(unittest.TestCase):
    def test_chebyshev_margin_enforced(self):
        # 2x1 板上两个单格零件相距 1（Chebyshev），违反一格边距
        res = solve(Board(2, 1), [Piece(MONO), Piece(MONO)])
        self.assertEqual(res.status, STATUS_INFEASIBLE)

    def test_distance_two_ok(self):
        # 3x1 板上两单格相距 2，合法；字典序最小为 x=0 与 x=2
        board = Board(3, 1)
        pieces = [Piece(MONO), Piece(MONO)]
        res = solve(board, pieces)
        self.assertEqual(res.status, STATUS_OPTIMAL)
        self.assertEqual(res.height, 1)
        assert_valid(self, board, pieces, res)
        self.assertEqual([p.x for p in res.placements], [0, 2])

    def test_diagonal_distance_one_rejected(self):
        # 3x3 板：单格 + 2x2 方块，若边距按 0 计算可放 (2,0)/(0,2)，
        # Chebyshev 一格边距下整块板放不下两者
        res = solve(Board(3, 3), [Piece(MONO), Piece(SQUARE2)])
        self.assertEqual(res.status, STATUS_INFEASIBLE)

    def test_margin_not_counted_outside_board(self):
        # 1x3 板两个单格：y=0 与 y=2；顶部膨胀越出板外不计高度
        board = Board(1, 3)
        pieces = [Piece(MONO), Piece(MONO)]
        res = solve(board, pieces)
        self.assertEqual(res.status, STATUS_OPTIMAL)
        self.assertEqual(res.height, 3)
        assert_valid(self, board, pieces, res)

    def test_forbidden_adjacency_allowed(self):
        # 禁放格只禁占用，不禁边距覆盖：单格可贴禁放格放置
        board = Board(3, 1, forbidden={(1, 0)})
        pieces = [Piece(MONO)]
        res = solve(board, pieces)
        self.assertEqual(res.status, STATUS_OPTIMAL)
        self.assertEqual((res.placements[0].x, res.placements[0].y), (0, 0))
        assert_valid(self, board, pieces, res)


class TestOrientationPolicy(unittest.TestCase):
    def test_rotation_only_fit(self):
        # L 三连块仅允许 0/90 度；禁放格 (0,1) 使 0 度放不下，必须旋转 90
        board = Board(2, 2, forbidden={(0, 1)})
        pieces = [Piece(L3, allowed_rotations=(0, 90))]
        res = solve(board, pieces)
        self.assertEqual(res.status, STATUS_OPTIMAL)
        self.assertEqual(res.placements[0].rotation, 90)
        self.assertFalse(res.placements[0].mirrored)
        self.assertEqual(res.rotation_count, 1)
        assert_valid(self, board, pieces, res)

    def test_rotation_disabled_stays_put(self):
        # 同一形状禁用旋转时该布局不可行
        board = Board(2, 2, forbidden={(0, 1)})
        pieces = [Piece(L3, allowed_rotations=(0,))]
        res = solve(board, pieces)
        self.assertEqual(res.status, STATUS_INFEASIBLE)

    def test_mirror_only_fit(self):
        # 禁放格 (0,1)：未镜像 L 被挡，仅镜像后可放；镜像不计旋转次数
        board = Board(2, 2, forbidden={(0, 1)})
        pieces = [Piece(L3, allowed_rotations=(0,), allow_mirror=True)]
        res = solve(board, pieces)
        self.assertEqual(res.status, STATUS_OPTIMAL)
        self.assertTrue(res.placements[0].mirrored)
        self.assertEqual(res.placements[0].rotation, 0)
        self.assertEqual(res.rotation_count, 0)
        assert_valid(self, board, pieces, res)

    def test_mirror_disabled_infeasible(self):
        board = Board(2, 2, forbidden={(0, 1)})
        pieces = [Piece(L3, allowed_rotations=(0,), allow_mirror=False)]
        res = solve(board, pieces)
        self.assertEqual(res.status, STATUS_INFEASIBLE)

    def test_zero_rotation_preferred(self):
        # 0 度即可放时不得使用旋转（旋转次数第二目标）
        board = Board(2, 2, forbidden={(1, 1)})
        pieces = [Piece(L3, allowed_rotations=(0, 90, 180, 270))]
        res = solve(board, pieces)
        self.assertEqual(res.status, STATUS_OPTIMAL)
        self.assertEqual(res.rotation_count, 0)
        self.assertEqual(res.placements[0].rotation, 0)


class TestHoleAndDuplicatePieces(unittest.TestCase):
    def test_ring_piece_alone(self):
        board = Board(3, 3)
        pieces = [Piece(RING, allowed_rotations=(0,))]
        res = solve(board, pieces)
        self.assertEqual(res.status, STATUS_OPTIMAL)
        self.assertEqual(res.height, 3)
        assert_valid(self, board, pieces, res)

    def test_ring_hole_unusable_by_monomino(self):
        # 环的孔与环格 Chebyshev 距离为 1，单格无法嵌入孔中
        res = solve(Board(3, 3), [Piece(RING, allowed_rotations=(0,)),
                                  Piece(MONO)])
        self.assertEqual(res.status, STATUS_INFEASIBLE)

    def test_duplicate_shapes_deduplicated(self):
        # 3 个相同横放多米诺在 8x1 板上：x = 0, 3, 6（等形对称去重后
        # 仍得到字典序最小 labeled 布局）
        board = Board(8, 1)
        pieces = [Piece(DOMINO, allowed_rotations=(0,)) for _ in range(3)]
        res = solve(board, pieces)
        self.assertEqual(res.status, STATUS_OPTIMAL)
        self.assertEqual(res.height, 1)
        self.assertEqual([p.x for p in res.placements], [0, 3, 6])
        assert_valid(self, board, pieces, res)

    def test_duplicate_shapes_infeasible(self):
        # 3 个多米诺在 5x1 板上只能放 2 个
        pieces = [Piece(DOMINO, allowed_rotations=(0,)) for _ in range(3)]
        res = solve(Board(5, 1), pieces)
        self.assertEqual(res.status, STATUS_INFEASIBLE)


class TestStatusSemantics(unittest.TestCase):
    def test_infeasible_proven(self):
        # 两个 2x2 方块在 3x3 板上放不下（边距约束）
        res = solve(Board(3, 3), [Piece(SQUARE2), Piece(SQUARE2)])
        self.assertEqual(res.status, STATUS_INFEASIBLE)
        self.assertIsNone(res.placements)
        self.assertIsNone(res.height)
        self.assertEqual(res.lower_bound_height, 4)  # 板高 3 + 1

    def test_infeasible_by_area(self):
        # 5 格零件放不进 2x2 板
        pento = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0))
        res = solve(Board(2, 2), [Piece(pento)])
        self.assertEqual(res.status, STATUS_INFEASIBLE)

    def test_budget_exhausted_distinct_from_infeasible(self):
        board = Board(2, 2)
        pieces = [Piece(MONO)]
        res = solve(board, pieces, budget=1)
        self.assertEqual(res.status, STATUS_BUDGET_EXHAUSTED)
        self.assertIsNone(res.placements)
        self.assertGreaterEqual(res.lower_bound_height, 1)
        # 同一问题给足预算则可证最优：两种状态必须分开
        res2 = solve(board, pieces, budget=100)
        self.assertEqual(res2.status, STATUS_OPTIMAL)

    def test_budget_exhausted_reports_best_known(self):
        # 预算足够找到某高度可行解、但不足以完成全部证明时，
        # 必须报告当前最优已知布局且不得声称最优
        board = Board(6, 4)
        pieces = [Piece(RING), Piece(DOMINO), Piece(DOMINO)]
        full = solve(board, pieces)
        self.assertEqual(full.status, STATUS_OPTIMAL)
        # 逐步压缩预算，应观察到 budget_exhausted 或 optimal，绝不出现
        # “高度更小却被证最优”的假最优
        for budget in (full.nodes - 1, max(full.nodes // 2, 1), 5):
            res = solve(board, pieces, budget=budget)
            self.assertIn(res.status, (STATUS_OPTIMAL, STATUS_BUDGET_EXHAUSTED))
            if res.status == STATUS_BUDGET_EXHAUSTED and res.height is not None:
                self.assertGreaterEqual(res.height, full.height)
                assert_valid(self, board, pieces, res)

    def test_nodes_within_budget(self):
        board = Board(5, 5)
        pieces = [Piece(RING), Piece(MONO)]
        res = solve(board, pieces, budget=10_000)
        self.assertLessEqual(res.nodes, 10_000)

    def test_exhausted_best_known_is_always_valid(self):
        # 对多档预算扫描：任何带布局的耗尽结果都必须是合法布局，
        # 且高度不小于真实最优、不小于已证下界的报告区间关系成立。
        board = Board(6, 5)
        pieces = [Piece(RING), Piece(DOMINO), Piece(DOMINO)]
        full = solve(board, pieces, budget=1_000_000)
        self.assertEqual(full.status, STATUS_OPTIMAL)
        seen_exhausted = False
        for budget in range(1, 200):
            res = solve(board, pieces, budget=budget)
            self.assertIn(res.status,
                          (STATUS_OPTIMAL, STATUS_BUDGET_EXHAUSTED,
                           STATUS_INFEASIBLE))
            self.assertLessEqual(res.nodes, budget)
            if res.status == STATUS_BUDGET_EXHAUSTED:
                seen_exhausted = True
                self.assertGreaterEqual(res.lower_bound_height, 1)
                if res.height is not None:
                    self.assertGreaterEqual(res.height, full.height)
                    self.assertGreaterEqual(res.height,
                                            res.lower_bound_height)
                    assert_valid(self, board, pieces, res)
        self.assertTrue(seen_exhausted, "小预算扫描应实际触发预算耗尽")

    def test_single_domino_height_one(self):
        # 单个横放多米诺在 6x6 板上使用高度为 1
        board = Board(6, 6)
        pieces = [Piece(DOMINO)]
        res = solve(board, pieces, budget=100_000)
        self.assertEqual(res.status, STATUS_OPTIMAL)
        self.assertEqual(res.height, 1)


class TestTieBreak(unittest.TestCase):
    def test_lexicographic_layout(self):
        # 两个不同零件均有多处可放时取坐标字典序最小
        board = Board(4, 1)
        pieces = [Piece(MONO), Piece(MONO)]
        res = solve(board, pieces)
        self.assertEqual(res.status, STATUS_OPTIMAL)
        self.assertEqual(
            [(p.x, p.y) for p in res.placements], [(0, 0), (2, 0)]
        )


if __name__ == "__main__":
    unittest.main()
