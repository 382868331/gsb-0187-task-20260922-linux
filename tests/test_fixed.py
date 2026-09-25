"""固定放置约束(FixedPlacement)专项测试。

覆盖:
* 两块同形件只锁一块 —— 锁定身份不可与自由件交换;
* 固定件形成窄洞 —— 可放/不可放两种;
* 锁定旋转(及镜像)不合法 —— 搜索前直接拒绝;
* 所有件都固定 —— 无需搜索即最优;
* 高度目标含固定件最高占用; 旋转次数第二目标只计可移动件;
  平局按含固定件的完整布局字典序;
* 固定项之间重叠/边距冲突直接拒绝, 不先移开;
* 预算耗尽仍给可行上界与有效下界;
* 小板上与独立穷举参考(_reference.solve 的 fixed 参数)逐项一致;
* 独立膨胀占用检查: 整板范围无冲突(不依赖求解器自身结论)。
"""

import random
import unittest

from nesting2d import (
    Board,
    FixedPlacement,
    InvalidSpecError,
    Part,
    Placement,
    Status,
    oriented_cells,
    pack,
    render_layout,
    verify_layout,
)

import _reference


def monomino():
    return Part([(0, 0)])


def domino(rotations=(0, 90)):
    return Part([(0, 0), (1, 0)], rotations=rotations)


def vdomino():
    """归一化即竖直 2 格, rotations=(0,)。"""
    return Part([(0, 0), (0, 1)], rotations=(0,))


class TestFixedValidation(unittest.TestCase):
    """锁定项在任何搜索之前完成校验, 不合法直接拒绝。"""

    def test_locked_rotation_not_allowed(self):
        board = Board(3, 3)
        parts = [domino(rotations=(0,))]  # 只允许 0 度
        with self.assertRaises(InvalidSpecError) as cm:
            pack(board, parts, fixed=[FixedPlacement(0, 0, 0, 90)])
        self.assertIn("旋转角", str(cm.exception))
        self.assertIn("fixed[0]", str(cm.exception))

    def test_locked_rotation_illegal_value(self):
        board = Board(3, 3)
        parts = [domino()]
        with self.assertRaises(InvalidSpecError):
            pack(board, parts, fixed=[FixedPlacement(0, 0, 0, 45)])
        with self.assertRaises(TypeError):
            pack(board, parts, fixed=[FixedPlacement(0, 0, 0, True)])

    def test_locked_mirror_disallowed(self):
        board = Board(4, 4)
        ell = Part(
            [(0, 0), (1, 0), (0, 1)],
            rotations=(0, 90, 180, 270),
            allow_reflection=False,
        )
        with self.assertRaises(InvalidSpecError):
            pack(board, [ell], fixed=[FixedPlacement(0, 0, 0, 0, True)])
        # 全局开关关闭时也拒绝(即使零件本身允许)
        ell2 = Part(
            [(0, 0), (1, 0), (0, 1)],
            rotations=(0, 90, 180, 270),
            allow_reflection=True,
        )
        with self.assertRaises(InvalidSpecError):
            pack(
                board,
                [ell2],
                allow_reflection=False,
                fixed=[FixedPlacement(0, 0, 0, 0, True)],
            )

    def test_index_errors(self):
        board = Board(3, 3)
        parts = [monomino(), monomino()]
        with self.assertRaises(InvalidSpecError):
            pack(board, parts, fixed=[FixedPlacement(2, 0, 0)])  # 越界
        with self.assertRaises(InvalidSpecError):
            pack(board, parts, fixed=[FixedPlacement(-1, 0, 0)])
        with self.assertRaises(InvalidSpecError):
            pack(  # 重复锁定同一零件
                board,
                parts,
                fixed=[FixedPlacement(0, 0, 0), FixedPlacement(0, 2, 0)],
            )
        with self.assertRaises(TypeError):
            pack(board, parts, fixed=[(0, 0, 0)])  # 非 FixedPlacement

    def test_out_of_board_negative_forbidden(self):
        # 越界
        with self.assertRaises(InvalidSpecError):
            pack(
                Board(3, 3),
                [domino(rotations=(0,))],
                fixed=[FixedPlacement(0, 2, 0, 0)],
            )
        # 负锚点
        with self.assertRaises(InvalidSpecError):
            pack(Board(3, 3), [monomino()], fixed=[FixedPlacement(0, -1, 0)])
        # 覆盖禁放格
        with self.assertRaises(InvalidSpecError):
            pack(
                Board(3, 3, forbidden=[(1, 1)]),
                [monomino()],
                fixed=[FixedPlacement(0, 1, 1)],
            )

    def test_fixed_pair_overlap_and_margin_rejected(self):
        board = Board(5, 5)
        parts = [monomino(), monomino()]
        # 重叠
        with self.assertRaises(InvalidSpecError) as cm:
            pack(board, parts, fixed=[FixedPlacement(0, 0, 0), FixedPlacement(1, 0, 0)])
        self.assertIn("重叠", str(cm.exception))
        # Chebyshev 距离 < 2: 相邻与对角都必须拒绝
        for bx, by in ((1, 0), (0, 1), (1, 1)):
            with self.subTest(bx=bx, by=by):
                with self.assertRaises(InvalidSpecError):
                    pack(
                        board,
                        parts,
                        fixed=[FixedPlacement(0, 0, 0), FixedPlacement(1, bx, by)],
                    )

    def test_fixed_never_moved_for_better_solution(self):
        # 两单格在 4x1 板上: 自由最优字典序布局是 (0,0),(2,0)。
        # 锁定零件 0 于 (3,0): 必须保持 (3,0), 绝不能把它"优化"到 (0,0)。
        board = Board(4, 1)
        parts = [monomino(), monomino()]
        res = pack(board, parts, fixed=[FixedPlacement(0, 3, 0)])
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.layout[0], (3, 0, 0, False))
        self.assertEqual(res.layout[1], (0, 0, 0, False))


class TestTwoIdenticalOneLocked(unittest.TestCase):
    """两块同形件只锁一块: 去对称不得交换锁定身份。"""

    def test_locked_identity_preserved(self):
        board = Board(6, 1)
        # 三个完全相同的单格; 锁零件 0 在最右的 (4,0)
        parts = [monomino(), monomino(), monomino()]
        res = pack(board, parts, fixed=[FixedPlacement(0, 4, 0)])
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.height, 1)
        # 自由件 1/2 取剩余字典序最小位置, 但零件 0 必须留在 (4,0)
        self.assertEqual(
            res.layout,
            {0: (4, 0, 0, False), 1: (0, 0, 0, False), 2: (2, 0, 0, False)},
        )
        self.assertEqual(verify_layout(board, parts, res.placements, 1), [])

    def test_locked_and_free_not_grouped(self):
        # 同形但只锁一块: 5x1 板两单格, 锁 0 于 (2,0)。
        # 自由件合法位为距 (2,0) Chebyshev >=2 的 x=0 或 x=4, 取字典序
        # 最小的 (0,0)。若错误地交换锁定身份, 零件 0 会被报离 (2,0),
        # 锁定要求即被抹掉 —— 本断言钉住零件 0 必须留在 (2,0)。
        board = Board(5, 1)
        parts = [monomino(), monomino()]
        res = pack(board, parts, fixed=[FixedPlacement(0, 2, 0)])
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.layout, {0: (2, 0, 0, False), 1: (0, 0, 0, False)})


class TestNarrowHole(unittest.TestCase):
    """固定件形成窄洞: 独立膨胀后的剩余空间决定可行性。"""

    def test_hole_exactly_one_cell(self):
        # 5x3 板, 两道竖直 2 格墙锁在 x=0 与 x=4(底部)。
        # 两墙膨胀后仅剩 (2,0) 一格可放单格(洞宽恰好一格)。
        board = Board(5, 3)
        parts = [vdomino(), vdomino(), monomino()]
        res = pack(
            board,
            parts,
            fixed=[FixedPlacement(0, 0, 0, 0), FixedPlacement(1, 4, 0, 0)],
        )
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.height, 2)
        self.assertEqual(res.layout[2], (2, 0, 0, False))
        self.assertEqual(verify_layout(board, parts, res.placements, res.height), [])
        art = render_layout(board, parts, res.placements, res.height)
        self.assertEqual(art.splitlines(), ["0...1", "0.2.1"])

    def test_hole_too_narrow_infeasible(self):
        # 墙锁在 x=0 与 x=3: 膨胀区在全部三行覆盖整个板宽 => 无洞。
        board = Board(5, 3)
        parts = [vdomino(), vdomino(), monomino()]
        res = pack(
            board,
            parts,
            fixed=[FixedPlacement(0, 0, 0, 0), FixedPlacement(1, 3, 0, 0)],
        )
        self.assertEqual(res.status, Status.INFEASIBLE)
        self.assertIsNone(res.placements)
        self.assertEqual(res.lower_bound, board.height + 1)

    def test_whole_board_independent_dilation_check(self):
        # 不使用求解器/校验器自身结论: 独立逐格 Chebyshev 检查,
        # 在**整板**范围内确认固定墙与自由件两两距离 >= 2 且不覆盖禁放格。
        board = Board(5, 3)
        parts = [vdomino(), vdomino(), monomino()]
        fixed = [FixedPlacement(0, 0, 0, 0), FixedPlacement(1, 4, 0, 0)]
        res = pack(board, parts, fixed=fixed)

        def cells_of(pl: Placement):
            shape = oriented_cells(
                parts[pl.part_index].cells, pl.rotation_degrees, pl.mirrored
            )
            return frozenset((x + pl.x, y + pl.y) for x, y in shape)

        occupied = [cells_of(pl) for pl in res.placements]
        # 全部格子在整板范围内且不碰禁放格
        for cells in occupied:
            for x, y in cells:
                self.assertTrue(0 <= x < board.width and 0 <= y < board.height)
                self.assertNotIn((x, y), board.forbidden)
        # 两两独立 Chebyshev 距离 >= 2
        for i in range(len(occupied)):
            for j in range(i + 1, len(occupied)):
                dists = [
                    max(abs(ax - bx), abs(ay - by))
                    for ax, ay in occupied[i]
                    for bx, by in occupied[j]
                ]
                self.assertGreaterEqual(min(dists), 2, (i, j, occupied))


class TestAllFixed(unittest.TestCase):
    """所有件都固定: 校验通过即唯一布局, 直接报最优、零搜索。"""

    def test_all_fixed_is_optimal_without_search(self):
        board = Board(5, 5)
        parts = [monomino(), monomino()]
        res = pack(
            board,
            parts,
            fixed=[FixedPlacement(0, 0, 3), FixedPlacement(1, 2, 3)],
        )
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.height, 4)  # 最高占用 y=3 -> 高度 4
        self.assertEqual(res.rotation_count, 0)
        self.assertEqual(res.lower_bound, 4)
        self.assertEqual(res.nodes, 0)  # 没有消耗任何搜索节点
        self.assertEqual(
            res.layout, {0: (0, 3, 0, False), 1: (2, 3, 0, False)}
        )

    def test_all_fixed_conflict_still_rejected(self):
        board = Board(5, 5)
        parts = [monomino(), monomino()]
        with self.assertRaises(InvalidSpecError):
            pack(
                board,
                parts,
                fixed=[FixedPlacement(0, 0, 0), FixedPlacement(1, 0, 0)],
            )


class TestObjectivesWithFixed(unittest.TestCase):
    """高度含固定件; 旋转次数只计可移动件; 平局按完整布局字典序。"""

    def test_height_includes_fixed_top(self):
        board = Board(5, 5)
        parts = [monomino(), monomino()]
        res = pack(board, parts, fixed=[FixedPlacement(0, 0, 3)])
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.height, 4)
        self.assertEqual(res.lower_bound, 4)
        # 自由件不会被放到比固定件更高的位置(无必要)
        self.assertLessEqual(res.layout[1][1], 3)

    def test_rotation_count_excludes_fixed(self):
        # 零件0(水平多米诺)锁定为 90 度: 它的旋转不计入第二目标。
        board = Board(4, 4)
        parts = [
            Part([(0, 0), (1, 0)], rotations=(0, 90)),
            monomino(),
        ]
        res = pack(board, parts, fixed=[FixedPlacement(0, 0, 0, 90)])
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.rotation_count, 0)  # 单格无旋转; 锁定件不计
        self.assertEqual(res.layout[0][2], 90)

    def test_rotation_count_still_minimized_among_movable(self):
        # 6x2: 两自由多米诺 + 一个锁定单格占住 (5,1)(膨胀封掉 x>=4)。
        # 高度 2 下两竖放要 2 次旋转; 一横一竖(竖 x=0, 横 (2,0)-(3,0))
        # 只需 1 次且与锁定单格保持边距 => 次目标应为 1。
        board = Board(6, 2)
        parts = [domino(), domino(), monomino()]
        res = pack(board, parts, fixed=[FixedPlacement(2, 5, 1)])
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.height, 2)
        self.assertEqual(res.rotation_count, 1)
        self.assertEqual(verify_layout(board, parts, res.placements, 2), [])

    def test_lex_break_on_full_layout(self):
        # 零件0锁定, 零件1/2 为自由单格。自由件的摆位应使**完整布局**
        # (按零件下标 0,1,2)字典序最小, 而不是只看自由件的相对顺序。
        board = Board(6, 2)
        parts = [monomino(), monomino(), monomino()]
        res = pack(board, parts, fixed=[FixedPlacement(0, 4, 1)])
        self.assertEqual(res.status, Status.OPTIMAL)
        # 固定条目参与完整布局键; 自由件取最低最左可用格
        self.assertEqual(res.layout[0], (4, 1, 0, False))
        self.assertEqual(res.layout[1], (0, 0, 0, False))
        self.assertEqual(res.layout[2], (2, 0, 0, False))


class TestFixedBudget(unittest.TestCase):
    """预算耗尽: 仍给可行上界(最优已知)与有效下界, 不假装最优。"""

    def test_exhausted_reports_incumbent_and_lb(self):
        board = Board(6, 6)
        parts = [
            monomino(),  # 0: 锁定在高处, 抬高高度下界
            domino(),
            domino(),
            domino(),
        ]
        fixed = [FixedPlacement(0, 5, 5)]
        res = pack(board, parts, budget=2, fixed=fixed)
        self.assertEqual(res.status, Status.BUDGET_EXHAUSTED)
        # 预算太小时可能尚无可行解; 下界必须有效(>= 固定件高度下界)
        self.assertGreaterEqual(res.lower_bound, 6)
        self.assertLessEqual(res.lower_bound, board.height + 1)
        # 给足预算: 高度必须被固定件顶到 6
        full = pack(board, parts, fixed=fixed)
        self.assertEqual(full.status, Status.OPTIMAL)
        self.assertEqual(full.height, 6)
        self.assertEqual(full.layout[0], (5, 5, 0, False))

    def test_exhausted_with_incumbent_keeps_fixed(self):
        board = Board(5, 5)
        parts = [domino(), domino(), domino(), monomino()]
        fixed = [FixedPlacement(3, 4, 0)]
        res = pack(board, parts, budget=20, fixed=fixed)
        if res.height is not None:
            # 一旦报告了最优已知解, 其中锁定件必须仍在锁定位置
            self.assertEqual(res.layout[3], (4, 0, 0, False))
            self.assertEqual(
                verify_layout(board, parts, res.placements, res.height), []
            )
            # 报告高度是最优高度的有效上界
            full = pack(board, parts, fixed=fixed)
            self.assertGreaterEqual(res.height, full.height)
            self.assertLessEqual(res.lower_bound, full.height)
        else:
            self.assertIsNone(res.placements)


def _fixed_ref_arg(fixed):
    return [(fp.part_index, fp.x, fp.y, fp.rotation_degrees, fp.mirrored)
            for fp in fixed]


class TestFixedReferenceCrossCheck(unittest.TestCase):
    """小板上锁定布局与独立穷举参考逐项一致(高度/旋转/完整字典序布局)。"""

    def check(self, board, parts, fixed, allow_reflection=True):
        res = pack(board, parts, allow_reflection=allow_reflection, fixed=fixed)
        ref = _reference.solve(
            board,
            parts,
            allow_reflection=allow_reflection,
            fixed=_fixed_ref_arg(fixed),
        )
        if ref is None:
            self.assertEqual(res.status, Status.INFEASIBLE)
            return
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.height, ref[0])
        self.assertEqual(res.rotation_count, ref[1])
        got = tuple(res.layout[i] for i in sorted(res.layout))
        self.assertEqual(got, ref[3])
        self.assertEqual(
            verify_layout(board, parts, res.placements, res.height), []
        )

    def test_handcrafted_fixed_cases(self):
        # 两同形件锁一块
        self.check(
            Board(6, 1),
            [monomino(), monomino(), monomino()],
            [FixedPlacement(0, 4, 0)],
        )
        # 窄洞恰好一格里放单格
        self.check(
            Board(5, 3),
            [vdomino(), vdomino(), monomino()],
            [FixedPlacement(0, 0, 0, 0), FixedPlacement(1, 4, 0, 0)],
        )
        # 窄洞封死(参考也判不可行)
        board = Board(5, 3)
        parts = [vdomino(), vdomino(), monomino()]
        fixed = [FixedPlacement(0, 0, 0, 0), FixedPlacement(1, 3, 0, 0)]
        res = pack(board, parts, fixed=fixed)
        self.assertEqual(res.status, Status.INFEASIBLE)
        self.assertIsNone(
            _reference.solve(board, parts, fixed=_fixed_ref_arg(fixed))
        )
        # 固定件在高处 + 自由多米诺
        self.check(
            Board(5, 5),
            [monomino(), domino(), domino()],
            [FixedPlacement(0, 0, 3)],
        )
        # 全部固定
        self.check(
            Board(5, 5),
            [monomino(), monomino()],
            [FixedPlacement(0, 0, 3), FixedPlacement(1, 2, 3)],
        )
        # 锁定 90 度(参考同样不计其旋转)
        self.check(
            Board(4, 4),
            [Part([(0, 0), (1, 0)], rotations=(0, 90)), monomino()],
            [FixedPlacement(0, 0, 0, 90)],
        )

    def test_random_small_instances_with_fixed(self):
        rng = random.Random(20260925)

        def random_shape():
            cells = {(0, 0)}
            target = rng.randint(1, 5)
            while len(cells) < target:
                x, y = rng.choice(sorted(cells))
                nb = rng.choice([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
                cells.add(nb)
            return sorted(cells)

        for trial in range(60):
            w = rng.randint(3, 5)
            h = rng.randint(3, 5)
            n = rng.randint(1, 4)
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
                if rng.random() < 0.08
            ]
            board = Board(w, h, forbidden=forbidden)
            allow_reflection = rng.random() < 0.5
            # 随机挑一部分零件, 给每个锁定件从其合法放置中随机选一个;
            # 锁定候选之间可能冲突 —— 与参考统一: 冲突时核心应拒绝,
            # 跳过该子例(拒绝语义由专门测试覆盖)。
            lock_idx = [i for i in range(n) if rng.random() < 0.4]
            fixed = []
            conflict = False
            used_cells = []
            for i in lock_idx:
                rots = [r for r in parts[i].rotations]
                r = rng.choice(rots)
                mir = (
                    allow_reflection
                    and parts[i].allow_reflection
                    and rng.random() < 0.3
                )
                shape = oriented_cells(parts[i].cells, r, mir)
                sw = max(x for x, _ in shape) + 1
                sh = max(y for _, y in shape) + 1
                if sw > w or sh > h:
                    conflict = True
                    break
                x = rng.randrange(w - sw + 1)
                y = rng.randrange(h - sh + 1)
                cells = {(cx + x, cy + y) for cx, cy in shape}
                if cells & set(forbidden):
                    conflict = True
                    break
                ok = True
                for prev in used_cells:
                    if any(
                        max(abs(ax - bx), abs(ay - by)) < 2
                        for ax, ay in cells
                        for bx, by in prev
                    ):
                        ok = False
                if not ok:
                    conflict = True
                    break
                used_cells.append(cells)
                fixed.append(FixedPlacement(i, x, y, r, mir))
            if conflict:
                continue
            with self.subTest(trial=trial, w=w, h=h, n=n, nfix=len(fixed)):
                self.check(board, parts, fixed, allow_reflection)


class TestVerifyFixed(unittest.TestCase):
    """verify_layout 的 fixed= 一致性核对。"""

    def test_pack_result_respects_fixed(self):
        board = Board(6, 1)
        parts = [monomino(), monomino(), monomino()]
        fixed = [FixedPlacement(0, 4, 0)]
        res = pack(board, parts, fixed=fixed)
        self.assertEqual(
            verify_layout(board, parts, res.placements, res.height, fixed=fixed),
            [],
        )

    def test_verify_detects_moved_locked_part(self):
        board = Board(4, 1)
        parts = [monomino(), monomino()]
        fixed = [FixedPlacement(0, 3, 0)]
        res = pack(board, parts, fixed=fixed)
        # 手工把锁定件挪到别处: 几何仍合法, 但违反锁定要求
        tampered = [
            Placement(0, 0, 0, 0, False),
            Placement(1, 2, 0, 0, False),
        ]
        problems = verify_layout(board, parts, tampered, 1, fixed=fixed)
        self.assertTrue(any("锁定" in p for p in problems), problems)
        # 不传 fixed 时只做几何校验, 不报锁定问题
        self.assertEqual(verify_layout(board, parts, tampered, 1), [])


if __name__ == "__main__":
    unittest.main()
