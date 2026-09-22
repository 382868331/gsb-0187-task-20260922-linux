"""输入校验与错误语义测试。"""

import math
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import nesting
from nesting import Board, NestingError, Piece, solve


class TestPieceValidation(unittest.TestCase):
    def test_empty_cells_rejected(self):
        with self.assertRaises(NestingError):
            Piece(cells=())

    def test_duplicate_cells_rejected(self):
        with self.assertRaises(NestingError):
            Piece(cells=((0, 0), (0, 0)))

    def test_disconnected_cells_rejected(self):
        with self.assertRaises(NestingError) as ctx:
            Piece(cells=((0, 0), (2, 0)))
        self.assertIn("cells", str(ctx.exception))

    def test_diagonal_not_connected(self):
        with self.assertRaises(NestingError):
            Piece(cells=((0, 0), (1, 1)))

    def test_bad_cell_type_rejected(self):
        with self.assertRaises(NestingError):
            Piece(cells=((0, 0), (1.5, 0)))
        with self.assertRaises(NestingError):
            Piece(cells=((0, 0), "x"))

    def test_nan_inf_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(NestingError):
                Piece(cells=((0, 0), (bad, 0)))

    def test_bool_coordinate_rejected(self):
        with self.assertRaises(NestingError):
            Piece(cells=((0, 0), (True, 0)))

    def test_rotation_values(self):
        with self.assertRaises(NestingError):
            Piece(cells=((0, 0),), allowed_rotations=(45,))
        with self.assertRaises(NestingError):
            Piece(cells=((0, 0),), allowed_rotations=())
        with self.assertRaises(NestingError):
            Piece(cells=((0, 0),), allowed_rotations=(0, 0))
        with self.assertRaises(NestingError):
            Piece(cells=((0, 0),), allowed_rotations=(True,))

    def test_mirror_must_be_bool(self):
        with self.assertRaises(NestingError):
            Piece(cells=((0, 0),), allow_mirror=1)
        with self.assertRaises(NestingError):
            Piece(cells=((0, 0),), allow_mirror="yes")

    def test_error_message_locatable(self):
        try:
            Piece(cells=((0, 0), (5, 5)))
        except NestingError as exc:
            self.assertTrue(str(exc).startswith("[Piece.cells]"))
        else:
            self.fail("应抛出 NestingError")


class TestBoardValidation(unittest.TestCase):
    def test_dimension_bounds(self):
        with self.assertRaises(NestingError):
            Board(0, 5)
        with self.assertRaises(NestingError):
            Board(5, 21)
        with self.assertRaises(NestingError):
            Board(3.5, 5)
        with self.assertRaises(NestingError):
            Board(True, 5)
        Board(20, 20)  # 上限边界合法

    def test_forbidden_out_of_range(self):
        with self.assertRaises(NestingError):
            Board(3, 3, forbidden={(3, 0)})
        with self.assertRaises(NestingError):
            Board(3, 3, forbidden={(0, -1)})


class TestSolveValidation(unittest.TestCase):
    def setUp(self):
        self.board = Board(3, 3)
        self.piece = Piece(cells=((0, 0),))

    def test_wrong_types(self):
        with self.assertRaises(NestingError):
            solve("not a board", [self.piece])
        with self.assertRaises(NestingError):
            solve(self.board, [self.piece, "not a piece"])
        with self.assertRaises(NestingError):
            solve(self.board, self.piece)  # 非序列

    def test_piece_count_bounds(self):
        with self.assertRaises(NestingError):
            solve(self.board, [])
        with self.assertRaises(NestingError):
            solve(self.board, [self.piece] * 13)

    def test_budget_validation(self):
        for bad in (0, -5, 1.5, True, math.nan):
            with self.assertRaises(NestingError):
                solve(self.board, [self.piece], budget=bad)

    def test_no_mutation_of_inputs(self):
        pieces = [self.piece]
        before = repr(pieces)
        solve(self.board, pieces)
        self.assertEqual(repr(pieces), before)


if __name__ == "__main__":
    unittest.main()
