"""切割段生成的测试套件。

期望值来源:
- 手算的小规模几何(单位正方形贴边、T 形部分重合、三角斜边、含孔环等);
- tests/_cut_reference.py 的独立中点覆盖参考(不导入 nesting2d)。
被测实现自身不参与期望值生成; 结构往返检查仅作补充。
"""

import random
import unittest
from fractions import Fraction

from nesting2d import (
    Board,
    InvalidSpecError,
    Part,
    Placement,
    PolyPart,
    PolyPlacement,
    Status,
    cut_segments,
    cut_segments_for_layout,
    oriented_cells,
    pack,
)

import _cut_reference as ref

F = Fraction
SQ = [(0, 0), (1, 0), (1, 1), (0, 1)]


def seg_tuples(plan):
    return [(s.p1, s.p2, s.parts, s.kerf) for s in plan.segments]


class TestPolyPartValidation(unittest.TestCase):
    """PolyPart 构造校验。"""

    def test_type_errors(self):
        with self.assertRaises(TypeError):
            PolyPart([(0, 0), (1.0, 0), (1, 1)])  # float 坐标
        with self.assertRaises(TypeError):
            PolyPart([(0, 0), (1, 0), (True, 1)])  # bool
        with self.assertRaises(TypeError):
            PolyPart([(0, 0), (1, 0), (1, float("nan"))])
        with self.assertRaises(TypeError):
            PolyPart([(0, 0), (1, 0), (1, float("inf"))])
        with self.assertRaises(TypeError):
            PolyPart("abcd")

    def test_domain_errors(self):
        with self.assertRaises(InvalidSpecError):
            PolyPart([])  # 空
        with self.assertRaises(InvalidSpecError):
            PolyPart([(0, 0), (1, 0)])  # 不足 3 点
        with self.assertRaises(InvalidSpecError):
            PolyPart([(0, 0), (1, 0), (1, 0)])  # 相邻重合
        # 自交(蝴蝶结)
        bowtie = [(0, 0), (2, 2), (2, 0), (0, 2)]
        with self.assertRaises(InvalidSpecError):
            PolyPart(bowtie)
        # 共线退化(中间点多余)
        with self.assertRaises(InvalidSpecError):
            PolyPart([(0, 0), (1, 0), (2, 0), (2, 1), (0, 1)])
        # 孔在外部
        outer = [(0, 0), (2, 0), (2, 2), (0, 2)]
        with self.assertRaises(InvalidSpecError):
            PolyPart(outer, holes=[[(5, 5), (6, 5), (6, 6)]])
        # 两孔交叠
        with self.assertRaises(InvalidSpecError):
            PolyPart(
                [(0, 0), (4, 0), (4, 4), (0, 4)],
                holes=[
                    [(1, 1), (2, 1), (2, 2), (1, 2)],
                    [(1, 1), (2, 1), (2, 2), (1, 2)],
                ],
            )
        # 超过孔数上限
        with self.assertRaises(InvalidSpecError):
            PolyPart(
                [(0, 0), (10, 0), (10, 10), (0, 10)],
                holes=[[(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]
                       for i in range(1, 6)],
            )
        # 超过顶点上限
        with self.assertRaises(InvalidSpecError):
            PolyPart([(0, 0)] + [(i, 0) for i in range(1, 33)])

    def test_orientation_normalized_and_repeated_endpoint_tolerated(self):
        cw = [(0, 0), (0, 1), (1, 1), (1, 0)]  # 顺时针
        p = PolyPart(cw)
        # 归一化后外轮廓为 CCW(两倍有向面积 > 0)
        x1, y1 = p.outer[0]
        x2, y2 = p.outer[1]
        x3, y3 = p.outer[2]
        area2 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        self.assertGreater(area2, 0)
        # 孔归一化为 CW
        outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
        hole_ccw = [(1, 1), (2, 1), (2, 2), (1, 2)]
        q = PolyPart(outer, holes=[hole_ccw])
        hx1, hy1 = q.holes[0][0]
        hx2, hy2 = q.holes[0][1]
        hx3, hy3 = q.holes[0][2]
        harea2 = (hx2 - hx1) * (hy3 - hy1) - (hy2 - hy1) * (hx3 - hx1)
        self.assertLess(harea2, 0)
        # 允许重复书写闭合点
        r = PolyPart([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        self.assertEqual(len(r.outer), 4)


class TestBasicSharedEdge(unittest.TestCase):
    def test_two_squares_share_full_edge(self):
        a = PolyPart(SQ)
        b = PolyPart(SQ)
        plan = cut_segments([a, b],
                            [PolyPlacement(0, 0, 0), PolyPlacement(1, 1, 0)])
        # 手算: 两个单位正方形共一条竖边 -> 4+4-1 = 7 段, 共边 1 段
        self.assertEqual(plan.raw_edge_count, 8)
        self.assertEqual(len(plan), 7)
        self.assertEqual(plan.shared_count, 1)
        shared = [s for s in plan.segments if s.shared]
        self.assertEqual(len(shared), 1)
        self.assertEqual(
            shared[0].p1, (F(1), F(0))
        )
        self.assertEqual(
            shared[0].p2, (F(1), F(1))
        )
        self.assertEqual(shared[0].parts, (0, 1))
        self.assertFalse(shared[0].kerf)
        # 非共边只保留单个零件
        for s in plan.segments:
            if not s.shared:
                self.assertEqual(len(s.parts), 1)

    def test_single_part_no_sharing(self):
        plan = cut_segments([PolyPart(SQ)], [PolyPlacement(0, 0, 0)])
        self.assertEqual(len(plan), 4)
        self.assertEqual(plan.shared_count, 0)
        self.assertTrue(all(not s.shared for s in plan.segments))

    def test_parts_ids_preserved_on_shared(self):
        # 三个正方形横排: 0-1 共边 x=1, 1-2 共边 x=2
        ps = [PolyPart(SQ) for _ in range(3)]
        pls = [PolyPlacement(i, i, 0) for i in range(3)]
        plan = cut_segments(ps, pls)
        shared = sorted(plan.segments,
                        key=lambda s: (s.p1[0], s.parts))
        shared = [s for s in shared if s.shared]
        self.assertEqual([s.parts for s in shared], [(0, 1), (1, 2)])
        self.assertEqual(len(plan), 10)  # 12 - 2

    def test_point_contact_not_merged(self):
        # 仅角接触: B 平移 (1,1)
        ps = [PolyPart(SQ), PolyPart(SQ)]
        plan = cut_segments(ps,
                            [PolyPlacement(0, 0, 0), PolyPlacement(1, 1, 1)])
        self.assertEqual(plan.shared_count, 0)
        self.assertEqual(len(plan), 8)


class TestPartialOverlap(unittest.TestCase):
    def test_t_junction_split(self):
        # A: [0,4]x[0,1]; B: 2x2 放在 (1,1)。
        # A 顶边 (0,1)-(4,1) 与 B 底边 (1,1)-(3,1) 部分重合。
        A = PolyPart([(0, 0), (4, 0), (4, 1), (0, 1)])
        B = PolyPart([(0, 0), (2, 0), (2, 2), (0, 2)])
        plan = cut_segments([A, B],
                            [PolyPlacement(0, 0, 0), PolyPlacement(1, 1, 1)])
        # y=1 的三段必须分开: 只A、共享、只A, 不能把整段错误去重
        y1 = [(s.p1[0], s.p2[0], s.parts)
              for s in plan.segments
              if s.p1[1] == 1 and s.p2[1] == 1]
        self.assertEqual(
            y1,
            [
                (F(0), F(1), (0,)),
                (F(1), F(3), (0, 1)),
                (F(3), F(4), (0,)),
            ],
        )
        self.assertEqual(plan.shared_count, 1)
        self.assertEqual(len(plan), 9)

    def test_partial_shared_vertical_edge(self):
        # A: 3x2 右边 x=3,y∈[0,2]; B: 1x1 放 (3,0), 只共下半边。
        A = PolyPart([(0, 0), (3, 0), (3, 2), (0, 2)])
        B = PolyPart(SQ)
        plan = cut_segments([A, B],
                            [PolyPlacement(0, 0, 0), PolyPlacement(1, 3, 0)])
        vert = [(s.p1[1], s.p2[1], s.parts)
                for s in plan.segments
                if s.p1[0] == 3 and s.p2[0] == 3]
        self.assertEqual(
            vert,
            [(F(0), F(1), (0, 1)), (F(1), F(2), (0,))],
        )

    def test_diagonal_shared_edge(self):
        # 两个直角三角形拼成一个方块, 斜边有理且完全重合
        t1 = PolyPart([(0, 0), (2, 0), (0, 2)])
        t2 = PolyPart([(2, 0), (2, 2), (0, 2)])
        plan = cut_segments([t1, t2],
                            [PolyPlacement(0, 0, 0), PolyPlacement(1, 0, 0)])
        self.assertEqual(len(plan), 5)
        diag = [s for s in plan.segments if s.shared]
        self.assertEqual(len(diag), 1)
        self.assertEqual(diag[0].p1, (F(0), F(2)))
        self.assertEqual(diag[0].p2, (F(2), F(0)))
        self.assertEqual(diag[0].parts, (0, 1))


class TestKerf(unittest.TestCase):
    def test_same_kerf_merges(self):
        ps = [PolyPart(SQ), PolyPart(SQ)]
        plan = cut_segments(ps,
                            [PolyPlacement(0, 0, 0), PolyPlacement(1, 1, 0)],
                            kerf=F(1, 8))
        shared = [s for s in plan.segments if s.shared]
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0].kerf, F(1, 8))

    def test_different_kerf_not_merged(self):
        ps = [PolyPart(SQ), PolyPart(SQ)]
        plan = cut_segments(
            ps,
            [PolyPlacement(0, 0, 0), PolyPlacement(1, 1, 0)],
            kerf=0,
            kerfs={0: F(1, 10), 1: F(2, 10)},
        )
        # 共边几何位置 x=1 上应有两段(不同 kerf), 都不是 shared
        x1 = [s for s in plan.segments
              if s.p1[0] == 1 and s.p2[0] == 1]
        self.assertEqual(len(x1), 2)
        self.assertEqual({s.kerf for s in x1}, {F(1, 10), F(1, 5)})
        self.assertTrue(all(not s.shared for s in x1))
        # 段总数: 几何 7 段中, 共边拆成 2 -> 8
        self.assertEqual(len(plan), 8)

    def test_bad_kerf(self):
        ps = [PolyPart(SQ)]
        with self.assertRaises(InvalidSpecError):
            cut_segments(ps, [PolyPlacement(0, 0, 0)], kerf=-1)
        with self.assertRaises(InvalidSpecError):
            cut_segments(ps, [PolyPlacement(0, 0, 0)], kerfs={0: -F(1, 2)})
        with self.assertRaises(TypeError):
            cut_segments(ps, [PolyPlacement(0, 0, 0)], kerf=0.5)
        with self.assertRaises(TypeError):
            cut_segments(ps, [PolyPlacement(0, 0, 0)], kerf=True)
        with self.assertRaises(TypeError):
            cut_segments(ps, [PolyPlacement(0, 0, 0)], kerfs="no")
        with self.assertRaises(TypeError):
            cut_segments(ps, [PolyPlacement(0, 0, 0)], kerfs={True: 1})
        with self.assertRaises(InvalidSpecError):  # 下标越界的 kerfs 键
            cut_segments(ps, [PolyPlacement(0, 0, 0)], kerfs={5: 1})


class TestHoles(unittest.TestCase):
    def test_part_with_hole_has_inner_contour(self):
        outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
        hole = [(1, 1), (2, 1), (2, 2), (1, 2)]
        p = PolyPart(outer, holes=[hole])
        plan = cut_segments([p], [PolyPlacement(0, 0, 0)])
        # 外 4 边 + 孔 4 边 = 8, 无共边
        self.assertEqual(plan.raw_edge_count, 8)
        self.assertEqual(len(plan), 8)
        self.assertEqual(plan.shared_count, 0)

    def test_inner_and_outer_never_merge(self):
        # 单个含孔零件: 孔边与外轮廓不重合(几何上分离)
        outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
        hole = [(1, 1), (3, 1), (3, 3), (1, 3)]
        p = PolyPart(outer, holes=[hole])
        plan = cut_segments([p], [PolyPlacement(0, 0, 0)])
        self.assertTrue(all(not s.shared for s in plan.segments))

    def test_part_nested_in_hole_shares_inner_contour(self):
        # A: 5x5 外方带 2x2 孔(左下角 (1,1)); B: 2x2 恰好嵌入孔中。
        # 四条孔边与 B 的外轮廓完全重合 -> 4 条共边, 且放置合法(不重叠)。
        A = PolyPart(
            [(0, 0), (5, 0), (5, 5), (0, 5)],
            holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
        )
        B = PolyPart([(0, 0), (2, 0), (2, 2), (0, 2)])
        plan = cut_segments([A, B],
                            [PolyPlacement(0, 0, 0), PolyPlacement(1, 1, 1)])
        self.assertEqual(plan.shared_count, 4)
        shared = sorted((s.p1, s.p2) for s in plan.segments if s.shared)
        self.assertEqual(
            shared,
            sorted([
                ((F(1), F(1)), (F(1), F(3))),
                ((F(3), F(1)), (F(3), F(3))),
                ((F(1), F(1)), (F(3), F(1))),
                ((F(1), F(3)), (F(3), F(3))),
            ]),
        )

    def test_two_ring_parts_share_outer_edge(self):
        # 两个 4x4 方框(带 2x2 孔)并排贴边
        outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
        hole = [(1, 1), (3, 1), (3, 3), (1, 3)]
        A = PolyPart(outer, holes=[hole])
        B = PolyPart(outer, holes=[hole])
        plan = cut_segments([A, B],
                            [PolyPlacement(0, 0, 0), PolyPlacement(1, 4, 0)])
        # 每条共边: 4 单位长的竖边被格点分成段后应重新连成一刀
        shared = [s for s in plan.segments if s.shared]
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0].p1, (F(4), F(0)))
        self.assertEqual(shared[0].p2, (F(4), F(4)))
        # 总段: raw 16(每框 4 外边+4 孔边), 共边两条合一条 -> 15
        self.assertEqual(plan.raw_edge_count, 16)
        self.assertEqual(len(plan), 15)


class TestPlacementErrors(unittest.TestCase):
    def setUp(self):
        self.sq = PolyPart([(0, 0), (2, 0), (2, 2), (0, 2)])

    def test_wrong_types(self):
        with self.assertRaises(TypeError):
            cut_segments("x", [PolyPlacement(0, 0, 0)])
        with self.assertRaises(TypeError):
            cut_segments([self.sq], [object()])
        with self.assertRaises(TypeError):
            cut_segments([object()], [PolyPlacement(0, 0, 0)])

    def test_count_and_index(self):
        with self.assertRaises(InvalidSpecError):  # 0 个零件
            cut_segments([], [])
        with self.assertRaises(InvalidSpecError):  # 放置数不匹配
            cut_segments([self.sq], [])
        with self.assertRaises(InvalidSpecError):  # 下标越界
            cut_segments([self.sq], [PolyPlacement(5, 0, 0)])
        with self.assertRaises(InvalidSpecError):  # 重复下标 + 缺件
            cut_segments(
                [self.sq, self.sq],
                [PolyPlacement(0, 0, 0), PolyPlacement(0, 3, 0)],
            )

    def test_overlap_rejected(self):
        # 正面积重叠(平移 (1,0))
        with self.assertRaises(InvalidSpecError):
            cut_segments(
                [self.sq, self.sq],
                [PolyPlacement(0, 0, 0), PolyPlacement(1, 1, 0)],
            )
        # 真穿越(平移 1/2)
        with self.assertRaises(InvalidSpecError):
            cut_segments(
                [self.sq, self.sq],
                [PolyPlacement(0, 0, 0), PolyPlacement(1, F(1, 2), 0)],
            )

    def test_negative_translation_rejected(self):
        with self.assertRaises(InvalidSpecError):
            cut_segments([self.sq], [PolyPlacement(0, -1, 0)])

    def test_too_many_parts(self):
        with self.assertRaises(InvalidSpecError):
            cut_segments(
                [PolyPart(SQ)] * 13,
                [PolyPlacement(i, i * 2, 0) for i in range(13)],
            )

    def test_no_partial_state_on_failure(self):
        # 失败只抛异常; 输入对象不被修改
        sq = PolyPart(SQ)
        before = sq.outer
        try:
            cut_segments(
                [sq, sq],
                [PolyPlacement(0, 0, 0), PolyPlacement(1, 1, 0)],
            )
        except InvalidSpecError:
            pass
        self.assertEqual(sq.outer, before)


class TestDeterminism(unittest.TestCase):
    def test_fixed_endpoint_and_segment_order(self):
        A = PolyPart([(0, 0), (4, 0), (4, 1), (0, 1)])
        B = PolyPart([(0, 0), (2, 0), (2, 2), (0, 2)])
        parts = [A, B]
        pls = [PolyPlacement(0, 0, 0), PolyPlacement(1, 1, 1)]
        p1 = seg_tuples(cut_segments(parts, pls))
        # 仅打乱 placements 列表顺序: 结果(几何 + 零件 ID 集合)应一致
        p2 = seg_tuples(cut_segments(parts, list(reversed(pls))))
        self.assertEqual(p1, p2)
        # 每段 p1 < p2
        for a, b, _parts, _k in p1:
            self.assertLess(a, b)
        # 整表有序
        self.assertEqual(p1, sorted(p1, key=lambda z: (z[0], z[1], z[3])))

    def test_repeated_calls_equal(self):
        ps = [PolyPart(SQ) for _ in range(3)]
        pls = [PolyPlacement(i, i, i % 2) for i in range(3)]
        r1 = cut_segments(ps, pls, kerf=F(1, 3))
        r2 = cut_segments(ps, pls, kerf=F(1, 3))
        self.assertEqual(seg_tuples(r1), seg_tuples(r2))


class TestReferenceCrossCheck(unittest.TestCase):
    """与独立中点覆盖参考交叉核对(轴对齐矩形排, 含部分贴边与不同 kerf)。"""

    def _rect_rings(self, w, h):
        # CCW 矩形
        return ([(0, 0), (w, 0), (w, h), (0, h)],)

    def test_handcrafted_row(self):
        rng = random.Random(20260924)
        for trial in range(60):
            n = rng.randint(1, 5)
            widths = [rng.randint(1, 3) for _ in range(n)]
            heights = [rng.randint(1, 4) for _ in range(n)]
            xs = []
            acc = 0
            for w in widths:
                xs.append(acc)
                acc += w
            # y 偏移随机, 但限制在使内部不重叠: x 开区间互不相交即可
            ys = [rng.randint(0, 2) for _ in range(n)]
            polys = [self._rect_rings(widths[i], heights[i]) for i in range(n)]
            trans = [(xs[i], ys[i]) for i in range(n)]
            kerf_choices = [F(0), F(1, 10), F(1, 4)]
            kerfs = [rng.choice(kerf_choices) for _ in range(n)]

            parts = [
                PolyPart([(0, 0), (widths[i], 0),
                          (widths[i], heights[i]), (0, heights[i])])
                for i in range(n)
            ]
            placements = [
                PolyPlacement(i, xs[i], ys[i]) for i in range(n)
            ]
            plan = cut_segments(parts, placements,
                                kerf=0, kerfs={i: kerfs[i] for i in range(n)})
            got = seg_tuples(plan)
            want = ref.reference_cut_segments(polys, trans, kerfs)
            self.assertEqual(got, want, f"trial={trial}")

    def test_diagonal_fraction_offset(self):
        # 斜边 + 有理分数偏移: 两三角形
        t1 = ([(0, 0), (2, 0), (0, 2)],)
        t2 = ([(2, 0), (2, 2), (0, 2)],)
        polys = [t1, t2]
        trans = [(F(0), F(0)), (F(0), F(0))]
        plan = cut_segments(
            [PolyPart(t1[0]), PolyPart(t2[0])],
            [PolyPlacement(0, 0, 0), PolyPlacement(1, 0, 0)],
            kerf=F(1, 3),
        )
        want = ref.reference_cut_segments(
            polys, trans, [F(1, 3), F(1, 3)]
        )
        self.assertEqual(seg_tuples(plan), want)

    def test_grid_ring_vs_reference(self):
        # 含孔多边形(方框)并排: 参考与实现一致
        outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
        hole = [(1, 1), (3, 1), (3, 3), (1, 3)]
        polys = [(outer, hole), (outer, hole)]
        trans = [(0, 0), (4, 0)]
        plan = cut_segments(
            [PolyPart(outer, holes=[hole]), PolyPart(outer, holes=[hole])],
            [PolyPlacement(0, 0, 0), PolyPlacement(1, 4, 0)],
        )
        self.assertEqual(
            seg_tuples(plan),
            ref.reference_cut_segments(polys, trans, [0, 0]),
        )


class TestGridLayoutCuts(unittest.TestCase):
    """cut_segments_for_layout: 从网格 pack 结果生成切割段。"""

    def test_pack_result_then_cut(self):
        board = Board(6, 3)
        parts = [
            Part([(0, 0), (1, 0), (0, 1), (1, 1)]),
            Part([(0, 0), (1, 0), (0, 1), (1, 1)]),
        ]
        res = pack(board, parts)
        self.assertEqual(res.status, Status.OPTIMAL)
        plan = cut_segments_for_layout(
            board, parts, res.placements, res.height
        )
        # pack 结果有一格边距, 不应有共边
        self.assertEqual(plan.shared_count, 0)
        self.assertEqual(plan.part_count, 2)

    def test_manual_adjacent_grid_parts_share_edge(self):
        # 两个 1x2 竖条(旋转 90)左右贴边, 共一条 y∈[0,2] 的竖边
        board = Board(3, 2)
        parts = [Part([(0, 0), (1, 0)], rotations=(0, 90))] * 2
        pls = [
            Placement(0, 0, 0, 90, False),
            Placement(1, 1, 0, 90, False),
        ]
        plan = cut_segments_for_layout(board, parts, pls, 2)
        shared = [s for s in plan.segments if s.shared]
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0].p1, (F(1), F(0)))
        self.assertEqual(shared[0].p2, (F(1), F(2)))
        self.assertEqual(shared[0].parts, (0, 1))
        # raw: 每个 1x2 竖条周长 6 条单位边, 两件 = 12; 共边(2 单位)合并
        self.assertEqual(plan.raw_edge_count, 12)
        self.assertEqual(len(plan), 7)

    def test_grid_ring_hole_preserved(self):
        # 3x3 环(8 格, 有孔): 轮廓边数 = 外 12(因格点)+孔?
        ring_cells = [(x, y) for x in range(3) for y in range(3)
                      if (x, y) != (1, 1)]
        board = Board(3, 3)
        parts = [Part(ring_cells, rotations=(0,))]
        res = pack(board, parts)
        plan = cut_segments_for_layout(
            board, parts, res.placements, res.height
        )
        # 外环 12 条单位边 + 孔环 4 条 = 16
        self.assertEqual(plan.raw_edge_count, 16)
        self.assertEqual(len(plan), 8)  # 共线连成 4 外边 + 4 孔边
        self.assertEqual(plan.shared_count, 0)

    def test_layout_validation_errors(self):
        board = Board(4, 2)
        parts = [Part([(0, 0), (1, 0), (0, 1), (1, 1)])] * 2
        # 越界高度
        with self.assertRaises(InvalidSpecError):
            cut_segments_for_layout(
                board, parts,
                [Placement(0, 0, 0, 0, False),
                 Placement(1, 2, 0, 0, False)],
                1,
            )
        # 覆盖禁放格
        b2 = Board(4, 2, forbidden=[(0, 0)])
        with self.assertRaises(InvalidSpecError):
            cut_segments_for_layout(
                b2, parts,
                [Placement(0, 0, 0, 0, False),
                 Placement(1, 2, 0, 0, False)],
                2,
            )
        # 重叠(无间距相邻在切割里合法, 但同格覆盖非法)
        with self.assertRaises(InvalidSpecError):
            cut_segments_for_layout(
                board, parts,
                [Placement(0, 0, 0, 0, False),
                 Placement(1, 1, 0, 0, False)],
                2,
            )
        # 不允许的旋转角
        p3 = [Part([(0, 0), (1, 0)], rotations=(0,))]
        with self.assertRaises(InvalidSpecError):
            cut_segments_for_layout(
                board, p3, [Placement(0, 0, 0, 90, False)], 2
            )
        # height 越界
        with self.assertRaises(InvalidSpecError):
            cut_segments_for_layout(board, parts,
                                    [Placement(0, 0, 0, 0, False)] * 2, 9)
        # 类型错误
        with self.assertRaises(TypeError):
            cut_segments_for_layout("b", parts, [], 2)

    def test_every_segment_is_real_boundary(self):
        # 结构检查: 每段都沿某零件的真实边界; 共边段必须同时落在两个
        # 零件的边界上。网格零件的边均为轴对齐整数边, 按单位区间核对。
        board = Board(8, 4)
        parts = [Part([(0, 0), (1, 0), (0, 1), (1, 1)]) for _ in range(3)]
        pls = [
            Placement(0, 0, 0, 0, False),
            Placement(1, 2, 0, 0, False),
            Placement(2, 4, 0, 0, False),
        ]
        plan = cut_segments_for_layout(board, parts, pls, 2)
        self.assertEqual(plan.shared_count, 2)

        def unit_boundary_edges(i):
            shape = oriented_cells(parts[i].cells, pls[i].rotation_degrees,
                                   pls[i].mirrored)
            cells = {(x + pls[i].x, y + pls[i].y) for x, y in shape}
            out = set()
            for x, y in cells:
                corners = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
                for a, b in ((corners[0], corners[1]),
                             (corners[1], corners[2]),
                             (corners[2], corners[3]),
                             (corners[3], corners[0])):
                    out.add(tuple(sorted((a, b))))
            return out

        all_edges = [unit_boundary_edges(i) for i in range(3)]

        def covered_by(seg, i):
            p1, p2 = seg.p1, seg.p2
            # 网格: 必为水平或竖直整数边
            if p1[1] == p2[1]:
                y = int(p1[1])
                for x in range(int(min(p1[0], p2[0])),
                               int(max(p1[0], p2[0]))):
                    if tuple(sorted(((x, y), (x + 1, y)))) not in all_edges[i]:
                        return False
                return p1[1] == int(p1[1])
            if p1[0] == p2[0]:
                x = int(p1[0])
                for y in range(int(min(p1[1], p2[1])),
                               int(max(p1[1], p2[1]))):
                    if tuple(sorted(((x, y), (x, y + 1)))) not in all_edges[i]:
                        return False
                return p1[0] == int(p1[0])
            return False

        for seg in plan.segments:
            for i in seg.parts:
                self.assertTrue(
                    covered_by(seg, i),
                    f"段 {seg.p1}-{seg.p2} 不是零件 {i} 的边界",
                )


class TestRationalAndMisc(unittest.TestCase):
    def test_fraction_coordinates(self):
        # 有理分数坐标的两个小方块在 x=1/2 处贴边
        half = F(1, 2)
        s1 = PolyPart([(0, 0), (half, 0), (half, half), (0, half)])
        s2 = PolyPart([(0, 0), (half, 0), (half, half), (0, half)])
        plan = cut_segments(
            [s1, s2],
            [PolyPlacement(0, 0, 0), PolyPlacement(1, half, 0)],
            kerf=F(1, 3),
        )
        shared = [s for s in plan.segments if s.shared]
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0].p1, (half, 0))
        self.assertEqual(shared[0].p2, (half, half))
        self.assertEqual(shared[0].kerf, F(1, 3))

    def test_non_adjacent_indices_share(self):
        # 零件 0 与 2 共边(下标不相邻), parts 必须保留 {0,2}
        ps = [PolyPart(SQ) for _ in range(3)]
        plan = cut_segments(
            ps,
            [PolyPlacement(0, 0, 0),
             PolyPlacement(2, 1, 0),   # 零件 2 与零件 0 贴边(x=1)
             PolyPlacement(1, 2, 0)],  # 零件 1 再与零件 2 贴边(x=2)
        )
        shared = sorted(s.parts for s in plan.segments if s.shared)
        self.assertEqual(shared, [(0, 2), (1, 2)])

    def test_cutplan_container_protocol(self):
        plan = cut_segments([PolyPart(SQ)], [PolyPlacement(0, 0, 0)])
        self.assertEqual(len(plan), 4)
        self.assertEqual(len(list(plan)), 4)


class TestGridContourPerimeter(unittest.TestCase):
    """网格零件轮廓: raw_edge_count 等于周长公式, 与旋转/镜像无关。"""

    def _perimeter(self, cells):
        cells = set(cells)
        p = 4 * len(cells)
        for x, y in cells:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (x + dx, y + dy) in cells:
                    p -= 1
        return p

    def test_l_shape_orientations(self):
        cells = [(0, 0), (1, 0), (2, 0), (0, 1)]
        board = Board(5, 5)
        for rot in (0, 90, 180, 270):
            for mir in (False, True):
                part = Part(cells, rotations=(0, 90, 180, 270))
                shape = oriented_cells(part.cells, rot, mir)
                maxx = max(x for x, _ in shape)
                maxy = max(y for _, y in shape)
                pls = [Placement(0, 0, 0, rot, mir)]
                plan = cut_segments_for_layout(
                    board, [part], pls, max(maxx, maxy) + 1
                )
                self.assertEqual(
                    plan.raw_edge_count, self._perimeter(cells),
                    f"rot={rot} mir={mir}",
                )

    def test_ring_perimeter(self):
        ring = [(x, y) for x in range(3) for y in range(3) if (x, y) != (1, 1)]
        part = Part(ring, rotations=(0,))
        plan = cut_segments_for_layout(
            Board(3, 3), [part], [Placement(0, 0, 0, 0, False)], 3
        )
        # 8 格环: 外周长 12 + 孔周长 4 = 16 条单位边
        self.assertEqual(plan.raw_edge_count, 16)


if __name__ == "__main__":
    unittest.main()
