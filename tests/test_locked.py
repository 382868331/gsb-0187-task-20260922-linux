"""nesting2d 固定放置(锁定)约束测试套件。

覆盖: 锁定项求解前验证(形状/允许旋转/板内/禁放格/固定项间冲突直接拒绝)、
两块同形件只锁一块(固定身份不可与自由件交换)、固定件形成窄洞、
锁定旋转不合法、所有件都固定、高度目标含锁定件最高占用、
旋转次数只统计可移动件、完整布局字典序破平局、预算耗尽语义。

期望值来源: 手工推导 + tests/_reference.py 独立穷举参考(不调用被测核心),
外加本文件内的独立膨胀占用检查(逐格 Chebyshev 膨胀, 整板无冲突)。
"""

import random
import unittest

from nesting2d import (
    Board,
    InvalidSpecError,
    LockedPlacement,
    Part,
    Status,
    pack,
    verify_layout,
)

import _reference


def monomino():
    return Part([(0, 0)])


def domino(rotations=(0, 90)):
    return Part([(0, 0), (1, 0)], rotations=rotations)


def square():
    return Part([(0, 0), (1, 0), (0, 1), (1, 1)], rotations=(0,))


def naive_full_check(board, parts, layout, locked):
    """独立膨胀占用检查: 逐格 Chebyshev 一格膨胀, 整板无冲突。

    不调用 nesting2d 的任何内部函数(方向变换用 _reference 的独立实现)。
    返回问题列表(空 = 通过)。
    """
    problems = []
    cells_of = {}
    for i, entry in layout.items():
        x, y, r, m = entry
        shape = _reference.transform(parts[i].cells, r, m)
        cells_of[i] = {(cx + x, cy + y) for cx, cy in shape}
    for i, cells in cells_of.items():
        for c in cells:
            if not (0 <= c[0] < board.width and 0 <= c[1] < board.height):
                problems.append(f"零件 {i}: 格子 {c} 出界")
            if c in board.forbidden:
                problems.append(f"零件 {i}: 格子 {c} 覆盖禁放格")
    idx = sorted(cells_of)
    for a in range(len(idx)):
        ca = cells_of[idx[a]]
        dil = {
            (x + dx, y + dy)
            for x, y in ca
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        }
        for b in range(a + 1, len(idx)):
            hit = dil & cells_of[idx[b]]
            if hit:
                problems.append(
                    f"零件 {idx[a]} 与零件 {idx[b]} 冲突(重叠或边距不足): {hit}"
                )
    for lk in locked:
        want = (lk.x, lk.y, lk.rotation_degrees, lk.mirrored)
        if layout.get(lk.part_index) != want:
            problems.append(
                f"零件 {lk.part_index} 的放置 {layout.get(lk.part_index)} "
                f"与锁定要求 {want} 不一致"
            )
    return problems


class TestLockedValidation(unittest.TestCase):
    """锁定项必须先验证; 违规(含固定项之间冲突)直接拒绝。"""

    def test_locked_rotation_not_allowed(self):
        # 锁定旋转不合法: 方块旋转 90 度形状不变, 但允许集不含 90 => 拒绝
        board = Board(4, 4)
        with self.assertRaises(InvalidSpecError):
            pack(board, [square()], locked=[LockedPlacement(0, 0, 0, 90)])
        with self.assertRaises(InvalidSpecError):
            pack(board, [domino()], locked=[LockedPlacement(0, 0, 0, 45)])

    def test_locked_mirror_rejected(self):
        board = Board(4, 4)
        ell = Part([(0, 0), (1, 0), (0, 1)], rotations=(0,), allow_reflection=False)
        with self.assertRaises(InvalidSpecError):
            pack(board, [ell], locked=[LockedPlacement(0, 0, 0, 0, True)])
        ell2 = Part([(0, 0), (1, 0), (0, 1)], rotations=(0,))
        with self.assertRaises(InvalidSpecError):
            pack(
                board, [ell2], allow_reflection=False,
                locked=[LockedPlacement(0, 0, 0, 0, True)],
            )

    def test_locked_out_of_board(self):
        board = Board(4, 4)
        with self.assertRaises(InvalidSpecError):
            pack(board, [monomino()], locked=[LockedPlacement(0, 4, 0, 0)])
        with self.assertRaises(InvalidSpecError):
            pack(board, [monomino()], locked=[LockedPlacement(0, 0, -1, 0)])
        with self.assertRaises(InvalidSpecError):
            pack(board, [domino()], locked=[LockedPlacement(0, 3, 0, 0)])

    def test_locked_covers_forbidden(self):
        board = Board(4, 4, forbidden=[(1, 1)])
        with self.assertRaises(InvalidSpecError):
            pack(board, [monomino()], locked=[LockedPlacement(0, 1, 1, 0)])

    def test_locked_conflict_between_fixed_rejected(self):
        # 固定项之间冲突直接拒绝, 不能先移开再给更好的解
        board = Board(6, 6)
        parts = [monomino(), monomino(), monomino()]
        # 重叠
        with self.assertRaises(InvalidSpecError):
            pack(board, parts, locked=[
                LockedPlacement(0, 0, 0, 0),
                LockedPlacement(1, 0, 0, 0),
            ])
        # 边距不足(相邻一格)
        with self.assertRaises(InvalidSpecError):
            pack(board, parts, locked=[
                LockedPlacement(0, 0, 0, 0),
                LockedPlacement(1, 1, 1, 0),
            ])

    def test_locked_index_errors(self):
        board = Board(4, 4)
        with self.assertRaises(InvalidSpecError):
            pack(board, [monomino()], locked=[LockedPlacement(1, 0, 0, 0)])
        with self.assertRaises(InvalidSpecError):
            pack(board, [monomino(), monomino()], locked=[
                LockedPlacement(0, 0, 0, 0),
                LockedPlacement(0, 2, 0, 0),  # 重复锁定同一零件
            ])

    def test_locked_type_errors(self):
        board = Board(4, 4)
        with self.assertRaises(TypeError):
            pack(board, [monomino()], locked="not-a-sequence")
        with self.assertRaises(TypeError):
            pack(board, [monomino()], locked=None)
        with self.assertRaises(TypeError):
            pack(board, [monomino()], locked=[object()])
        with self.assertRaises(TypeError):
            pack(board, [monomino()], locked=[LockedPlacement(True, 0, 0, 0)])
        with self.assertRaises(TypeError):
            pack(board, [monomino()], locked=[LockedPlacement(0, 1.5, 0, 0)])
        with self.assertRaises(TypeError):
            pack(board, [monomino()], locked=[LockedPlacement(0, 0, 0, 0, 1)])


class TestLockedBasics(unittest.TestCase):
    """锁定语义: 身份不可交换、窄洞、全固定、高度/旋转次数口径。"""

    def check_against_reference(self, board, parts, locked, allow_reflection=True):
        res = pack(board, parts, allow_reflection=allow_reflection, locked=locked)
        ref = _reference.solve(
            board, parts, allow_reflection=allow_reflection, locked=locked
        )
        self.assertIsNotNone(ref)
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.height, ref[0])
        self.assertEqual(res.rotation_count, ref[1])
        got = tuple(res.layout[i] for i in sorted(res.layout))
        self.assertEqual(got, ref[3])  # 完整布局字典序一致
        self.assertEqual(
            verify_layout(board, parts, res.placements, res.height, locked=locked),
            [],
        )
        self.assertEqual(naive_full_check(board, parts, res.layout, locked), [])
        return res

    def test_two_identical_shapes_only_one_locked(self):
        # 两块同形件只锁一块: 锁定身份不可与自由件交换
        board = Board(5, 3)
        parts = [square(), square()]
        locked = [LockedPlacement(1, 3, 1, 0)]
        res = self.check_against_reference(board, parts, locked)
        self.assertEqual(res.layout[1], (3, 1, 0, False))  # 锁定件原地不动
        self.assertEqual(res.layout[0], (0, 0, 0, False))
        self.assertEqual(res.height, 3)  # 含锁定件最高占用
        # 不锁定时最优高度为 2: 锁定确实改变了结果而不是被忽略
        free = pack(board, parts)
        self.assertEqual(free.height, 2)

    def test_locked_identity_not_swapped_with_free(self):
        # 两个相同单格, 锁零件1在 (2,0): 若对称破缺把固定身份与自由件交换,
        # 结果会把零件1报在 (0,0); 必须保持零件1在锁定位置
        board = Board(4, 2)
        parts = [monomino(), monomino()]
        locked = [LockedPlacement(1, 2, 0, 0)]
        res = self.check_against_reference(board, parts, locked)
        self.assertEqual(res.layout, {0: (0, 0, 0, False), 1: (2, 0, 0, False)})

    def test_locked_parts_form_narrow_pocket(self):
        # 固定件形成窄洞: 两个锁定多米诺之间的 2 宽口袋因边距不可用,
        # 自由多米诺只能上移到第 3 行
        board = Board(6, 4)
        parts = [domino(), domino(), domino()]
        locked = [
            LockedPlacement(0, 0, 0, 0),  # 占 (0,0),(1,0)
            LockedPlacement(1, 4, 0, 0),  # 占 (4,0),(5,0)
        ]
        res = self.check_against_reference(board, parts, locked)
        self.assertEqual(res.height, 3)
        self.assertEqual(res.rotation_count, 0)
        self.assertEqual(
            res.layout,
            {0: (0, 0, 0, False), 1: (4, 0, 0, False), 2: (0, 2, 0, False)},
        )

    def test_all_parts_locked(self):
        board = Board(4, 4)
        parts = [monomino(), monomino()]
        locked = [LockedPlacement(0, 0, 0, 0), LockedPlacement(1, 2, 0, 0)]
        res = pack(board, parts, locked=locked)
        self.assertEqual(res.status, Status.OPTIMAL)
        self.assertEqual(res.height, 1)
        self.assertEqual(res.rotation_count, 0)
        self.assertEqual(res.nodes, 0)
        self.assertEqual(res.lower_bound, 1)
        self.assertEqual(res.layout, {0: (0, 0, 0, False), 1: (2, 0, 0, False)})
        ref = _reference.solve(board, parts, locked=locked)
        self.assertEqual((ref[0], ref[1], ref[3]), (1, 0, ((0, 0, 0, False), (2, 0, 0, False))))

    def test_height_includes_locked_top(self):
        # 锁定件占据顶行: 自由件再低, 高度也必须覆盖锁定件
        board = Board(3, 6)
        parts = [monomino(), monomino()]
        locked = [LockedPlacement(0, 1, 5, 0)]
        res = self.check_against_reference(board, parts, locked)
        self.assertEqual(res.height, 6)
        self.assertEqual(res.layout[1], (0, 0, 0, False))

    def test_rotation_count_excludes_locked(self):
        # 锁定件旋转 90 度不计入旋转次数; 自由件不旋转 => 总计 0
        board = Board(4, 3)
        parts = [domino(), domino()]
        locked = [LockedPlacement(0, 0, 0, 90)]  # 竖放, 占 (0,0),(0,1)
        res = self.check_against_reference(board, parts, locked)
        self.assertEqual(res.layout[0], (0, 0, 90, False))
        self.assertEqual(res.rotation_count, 0)
        self.assertEqual(res.height, 2)

    def test_locked_infeasible_remaining(self):
        # 锁定后剩余空间放不下自由件 => 证明不可行(而非移动锁定件)
        board = Board(3, 3)
        parts = [monomino()] * 4
        # 锁一个单格在中心, 剩余格子在边距下最多再放 2 个 => 4 个不可行
        locked = [LockedPlacement(0, 1, 1, 0)]
        res = pack(board, parts, locked=locked)
        self.assertEqual(res.status, Status.INFEASIBLE)
        self.assertIsNone(res.placements)
        self.assertIsNone(_reference.solve(board, parts, locked=locked))


class TestLockedBudget(unittest.TestCase):
    """预算耗尽: 继续给出可行上界与有效下界。"""

    def test_tiny_budget_with_locked(self):
        board = Board(5, 5)
        parts = [domino(), domino(), domino()]
        locked = [LockedPlacement(0, 0, 0, 0)]
        res = pack(board, parts, budget=1, locked=locked)
        self.assertEqual(res.status, Status.BUDGET_EXHAUSTED)
        self.assertIsNone(res.height)
        full = pack(board, parts, locked=locked)
        self.assertEqual(full.status, Status.OPTIMAL)
        self.assertGreaterEqual(res.lower_bound, 1)
        self.assertLessEqual(res.lower_bound, full.height)  # 有效下界

    def test_exhausted_reports_feasible_upper_bound(self):
        board = Board(5, 5)
        parts = [domino(), domino(), domino()]
        locked = [LockedPlacement(0, 0, 0, 0)]
        full = pack(board, parts, locked=locked)
        found = None
        for b in range(1, 5000):
            r = pack(board, parts, budget=b, locked=locked)
            if r.status is Status.BUDGET_EXHAUSTED and r.height is not None:
                found = r
                break
            if r.status is Status.OPTIMAL:
                break
        self.assertIsNotNone(found, "应存在预算使贪心已成功但证明未完成")
        # 可行上界: 报告的布局真实合法, 高度不差于最优
        self.assertEqual(
            verify_layout(board, parts, found.placements, found.height,
                          locked=locked),
            [],
        )
        self.assertEqual(naive_full_check(board, parts, found.layout, locked), [])
        self.assertGreaterEqual(found.height, full.height)
        self.assertLessEqual(found.lower_bound, full.height)  # 有效下界


class TestLockedOptimalityCrossCheck(unittest.TestCase):
    """小板穷举未锁定放置, 与独立参考核对最优性。"""

    def check(self, board, parts, locked, allow_reflection=True):
        res = pack(board, parts, allow_reflection=allow_reflection, locked=locked)
        ref = _reference.solve(
            board, parts, allow_reflection=allow_reflection, locked=locked
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
            verify_layout(board, parts, res.placements, res.height, locked=locked),
            [],
        )
        self.assertEqual(naive_full_check(board, parts, res.layout, locked), [])

    def test_fixed_cases(self):
        # 同形三块锁一块 + 禁放格
        board = Board(5, 4, forbidden=[(2, 2)])
        parts = [domino(), domino(), domino()]
        self.check(board, parts, [LockedPlacement(2, 4, 0, 90)])
        # L 形 + 镜像
        board2 = Board(4, 4)
        ell = Part([(0, 0), (1, 0), (0, 1)], rotations=(0, 90, 180, 270))
        self.check(board2, [ell, ell], [LockedPlacement(0, 0, 0, 0)])
        self.check(
            board2, [ell, ell], [LockedPlacement(0, 0, 0, 90)],
            allow_reflection=False,
        )

    def test_random_locked_instances(self):
        rng = random.Random(20260925)

        def random_shape():
            cells = {(0, 0)}
            target = rng.randint(1, 4)
            while len(cells) < target:
                x, y = rng.choice(sorted(cells))
                nb = rng.choice([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
                cells.add(nb)
            return sorted(cells)

        for trial in range(30):
            w = rng.randint(3, 5)
            h = rng.randint(3, 5)
            n = rng.randint(2, 4)
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
            allow_reflection = rng.random() < 0.5
            base = pack(board, parts, allow_reflection=allow_reflection)
            if base.status is not Status.OPTIMAL:
                continue
            # 锁定子集取自一个可行解 => 锁定必然合法; 含全部锁定的情形
            lay = base.layout
            locked = [
                LockedPlacement(i, *lay[i][:2], lay[i][2], lay[i][3])
                for i in range(n)
                if rng.random() < 0.5
            ]
            with self.subTest(trial=trial, locked=len(locked)):
                self.check(board, parts, locked, allow_reflection)
                if locked:
                    # 锁定的是可行解的子集: 加锁后的最优高度不会更差于…
                    # (上界角度) 不优于原最优? 加锁可行解仍在 => 高度 <= 原高度
                    res = pack(
                        board, parts,
                        allow_reflection=allow_reflection, locked=locked,
                    )
                    self.assertLessEqual(res.height, base.height)


class TestVerifyLayoutLocked(unittest.TestCase):
    def test_verify_layout_locked_consistency(self):
        board = Board(5, 3)
        parts = [square(), square()]
        locked = [LockedPlacement(1, 3, 1, 0)]
        res = pack(board, parts, locked=locked)
        self.assertEqual(
            verify_layout(board, parts, res.placements, res.height, locked=locked),
            [],
        )
        # 篡改锁定要求 => 必须检出
        tampered = [LockedPlacement(1, 0, 0, 0)]
        problems = verify_layout(
            board, parts, res.placements, res.height, locked=tampered
        )
        self.assertTrue(any("锁定" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
