"""与独立穷举参考实现的交叉验证（小板全枚举）。

参考实现 tests/reference.py 不调用被测核心；本文件对确定性生成的
小案例逐一比对：状态、最小高度、最小旋转次数、字典序最小布局键。
"""

import os
import random
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import nesting  # noqa: F401  (确保被测包可导入)
from helpers import validate_layout
from nesting import Board, Piece, STATUS_INFEASIBLE, STATUS_OPTIMAL, solve
from reference import reference_solve

RING = tuple((x, y) for x in range(3) for y in range(3) if (x, y) != (1, 1))


def random_piece(rng, max_cells):
    """随机游走生成 4-连通格子集合。"""
    cells = {(0, 0)}
    target = rng.randint(1, max_cells)
    while len(cells) < target:
        x, y = rng.choice(tuple(cells))
        dx, dy = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        cells.add((x + dx, y + dy))
    rots = tuple(r for r in (0, 90, 180, 270) if rng.random() < 0.8) or (0,)
    return (tuple(sorted(cells)), rots, rng.random() < 0.4)


def core_lex_key(result):
    rotations = (0, 90, 180, 270)
    return tuple((p.x, p.y, rotations.index(p.rotation) * 2
                  + (1 if p.mirrored else 0)) for p in result.placements)


class TestCrossCheck(unittest.TestCase):
    def check_case(self, width, height, forbidden, piece_specs):
        board = Board(width, height, forbidden=frozenset(forbidden))
        pieces = [Piece(cells=c, allowed_rotations=r, allow_mirror=m)
                  for c, r, m in piece_specs]
        result = solve(board, pieces)
        ref = reference_solve(width, height, forbidden, piece_specs)

        self.assertEqual(
            result.status,
            STATUS_OPTIMAL if ref["status"] == "optimal" else STATUS_INFEASIBLE,
            f"状态不一致: core={result.status} ref={ref['status']} "
            f"case=({width},{height},{forbidden},{piece_specs})",
        )
        self.assertEqual(result.height, ref["height"])
        self.assertEqual(result.rotation_count, ref["rotation_count"])
        if ref["status"] == "optimal":
            self.assertEqual(core_lex_key(result), ref["lex_key"])
            err = validate_layout(board, pieces, result.placements, result.height)
            self.assertIsNone(err, err)

    def test_fixed_cases(self):
        # 含孔零件 + 禁放格
        self.check_case(4, 4, {(3, 3)}, [(RING, (0, 90), False)])
        # 重复形
        dom = ((0, 0), (1, 0))
        self.check_case(5, 2, set(), [(dom, (0, 90), False)] * 3)
        # 仅旋转可放
        l3 = ((0, 0), (1, 0), (0, 1))
        self.check_case(2, 2, {(0, 1)}, [(l3, (0, 90), False)])
        # 仅镜像可放
        self.check_case(2, 2, {(0, 1)}, [(l3, (0,), True)])
        # 不可行
        sq = ((0, 0), (1, 0), (0, 1), (1, 1))
        self.check_case(3, 3, set(), [(sq, (0,), False), (sq, (0,), False)])

    def test_random_small_cases(self):
        rng = random.Random(20260922)
        for case_no in range(40):
            width = rng.randint(2, 5)
            height = rng.randint(2, 5)
            n_forbidden = rng.randint(0, 3)
            all_cells = [(x, y) for x in range(width) for y in range(height)]
            forbidden = set(rng.sample(all_cells, n_forbidden))
            n_pieces = rng.randint(1, 4)
            specs = [random_piece(rng, 5) for _ in range(n_pieces)]
            with self.subTest(case=case_no, board=(width, height),
                              forbidden=forbidden, pieces=specs):
                self.check_case(width, height, forbidden, specs)


if __name__ == "__main__":
    unittest.main()
