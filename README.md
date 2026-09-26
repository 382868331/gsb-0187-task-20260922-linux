# nesting2d — 约束二维材料片的无重叠排样 + 整刀切割检查

离线制造几何工程用可复用模块: 在整数网格板材上无重叠排样连通格子零件,
求最小使用高度; 并对给定布局做**整刀(guillotine)可加工性检查**与最优
切割树生成。**纯 Python 标准库实现, 从零编写, 不依赖任何第三方包。**

- Python 版本: **3.14.7**(仅标准库; 安装 Python 后无需安装任何其他依赖)
- 平台: Linux 原生; 核心、测试、演示全部离线, 不需要网络/账号/数据库/Docker

## 环境准备(venv)

```bash
python3.14 -m venv .venv          # 任意 3.14.7 解释器均可
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python demo.py           # 排样演示
.venv/bin/python demo_iteration.py # 整刀切割演示
```

无第三方依赖, `pip install` 什么也不需要装; 也可以直接用系统
`python3.14` 代替 `.venv/bin/python`。

## 问题定义

- 板材: `width × height` 整数网格, 上限 **20 × 20**; 可含禁放格。
- 零件: 4-连通的格子集合, 每零件 ≤ **12** 格, 共 ≤ **12** 件。
- 每个零件可独立指定允许的旋转角度(`0/90/180/270` 的非空子集, 默认仅 `0`)
  与是否允许镜像(默认允许); `pack(..., allow_reflection=False)` 全局禁用镜像。
- **边距**: 任意两个不同零件的格子之间 Chebyshev 距离 ≥ 2
  (等价于每个零件按 Chebyshev 一格膨胀后互不相交)。
  **板外不计额外空间**: 零件可紧贴板材边缘; 膨胀掩码在板边界处裁剪。
  禁放格只禁止被零件格子覆盖, 不参与边距膨胀(可贴边、可落入零件孔洞)。
- **使用高度** `h`: 只使用板材底部 `h` 行(`0 ≤ y < h`), `1 ≤ h ≤ height`。

### 优化目标(按优先级)

1. 最小化使用高度 `h`;
2. 最小化旋转次数 `rotation_count` = 放置方向旋转角 ≠ 0 的零件个数
   (镜像不计入旋转次数);
3. 最小化布局坐标字典序: 按零件下标比较每个零件的
   `(x, y, (rotation_degrees, mirrored))`。

坐标系: `(x, y)`, `0 ≤ x < width`, `0 ≤ y < height`, y 向上。
`Placement(x, y)` 是该零件所取方向归一化形状(最小坐标为 `(0,0)`)的锚点。

## 接口

```python
from nesting2d import Board, Part, pack, Status, verify_layout, render_layout

board = Board(5, 5, forbidden=[(4, 3)])
parts = [
    Part([(0, 0), (1, 0), (2, 0), (3, 0)], rotations=(0, 90)),
    Part([(0, 0), (1, 0), (0, 1), (1, 1)]),          # 默认 rotations=(0,)
    Part([(0, 0), (1, 0), (0, 1), (1, 1)]),
]
res = pack(board, parts, budget=1_000_000, allow_reflection=True)
res.status          # Status.OPTIMAL / INFEASIBLE / BUDGET_EXHAUSTED
res.height          # 最优(或最优已知)高度; 无可行解为 None
res.placements      # tuple[Placement], 按零件下标排序
res.layout          # dict: part_index -> (x, y, rotation_degrees, mirrored)
res.rotation_count  # 旋转次数
res.lower_bound     # 已证明的下界(见下)
res.nodes, res.budget
```

辅助函数:

- `verify_layout(board, parts, placements, height) -> list[str]`:
  独立校验布局, 返回问题列表(空 = 合法)。
- `render_layout(board, parts, placements, height) -> str`:
  字符画(顶行在上; 零件 `0-9A-Za-z`, 禁放格 `#`, 空白 `.`)。
- `oriented_cells(cells, rotation_degrees, mirrored)`: 方向变换(与求解器同一代码路径)。

## 整刀(guillotine)切割检查

给定一个**已经排好的**布局(位置固定, 不重新优化排样), 判断能否只用
整刀把各零件从板材上切出来, 并生成最优切割树:

- 每一刀必须**贯穿当前子板**(从一条边切到对边), 切线只落在整数网格线上;
- 切线**不得穿过任何零件内部**(零件横跨切线 = 该零件在切线两侧有相邻格),
  但**可以沿零件边界走**(不同零件分居切线两侧不构成阻挡);
- 叶节点要么恰好包含**一件完整零件**, 要么是**废料**;
  **零宽废料不单独成片**(切线坐标严格在子板内部, 两个子板都非退化);
- 目标按优先级: **最少刀数 → 最短总切长 → 切割树坐标字典序**
  (切向 v<h, 再按切线坐标与子树递归比较)。

```python
from nesting2d import guillotine_cut, GuillotineStatus

res = guillotine_cut(board, parts, placements, budget=1_000_000)
res.status        # CUTTABLE / NOT_GUILLOTINEABLE / BUDGET_EXHAUSTED
res.cuts          # tuple[Cut]: 先序全部刀, 每刀带 axis/coord/rect(子板范围)/length
res.tree          # 切割树: CutBranch(cut, lower, upper) / CutLeaf(rect, part_index|None)
res.total_length  # 总切长(整数)
res.blocking      # 不可切时根子板全部候选切线的阻挡见证 tuple[BlockedCut]
res.nodes, res.budget
```

状态语义(预算耗尽与确定不可切严格分开):

| 状态 | 含义 | cuts/tree | blocking |
|---|---|---|---|
| `CUTTABLE` | 找到最优整刀方案(三级目标均已优化) | 最优方案 | `None` |
| `NOT_GUILLOTINEABLE` | **已证明**根子板无任何合法第一刀 | `None` | 每条候选切线的阻挡零件 |
| `BUDGET_EXHAUSTED` | 预算耗尽, **不下结论** | `None` | `None` |

布局合法性: 每个零件恰好放置一次、方向被允许、格子在板内、不覆盖
禁放格、互不重叠; 违反即抛 `InvalidSpecError`。注意这里**不**强制
零件间 Chebyshev 边距 —— 边距是排样阶段约束, 与切割无关; 紧贴的
无重叠布局照样可以做整刀检查。预算以**递归处理的不同子板矩形数**计
(记忆化去重后计一次)。

### 结果状态语义(预算耗尽与证明无解严格区分)

| 状态 | 含义 | height/placements | lower_bound |
|---|---|---|---|
| `OPTIMAL` | 高度与全部次序目标均已证明最优 | 最优解 | 等于 `height` |
| `INFEASIBLE` | 已证明任何高度(≤ 板高)都不可行 | `None` | `board.height + 1` |
| `BUDGET_EXHAUSTED` | 预算耗尽, **不假装最优** | 最优已知解(可能 `None`) | 已证明的下界 |

下界 = max(1, ⌈总面积/板宽⌉, 各零件最小包围盒高度的最大值)。
`budget` 以 DFS 节点数计, 所有高度的搜索共享同一预算; 耗尽时 `nodes`
为实际访问计数(可能等于 `budget + 1`, 触发中止的节点也计入)。

## 错误语义

不静默纠正任何非法输入, 错误消息含可定位字段路径:

- `TypeError`: 类型层面问题 —— 非 int 数值、**bool 充当整数**、
  **NaN/Infinity**、非 bool 的开关参数、错误容器类型等;
- `InvalidSpecError(ValueError)`: 取值域违规 —— 板尺寸超出 `[1, 20]`、
  零件为空/重复格/不连通/超过 12 格/超过 12 件、旋转角非 `0/90/180/270`、
  空旋转集、禁放格越界或重复、`budget < 1` 等。

`Part`/`Board` 在构造时(`__post_init__)完成全部校验并归一化为不可变
`frozenset`, 之后 `pack` 不会修改输入; 求解过程无可变全局状态,
失败不会留下部分变更(结果只通过返回值传递)。

## 运行

```bash
python -m unittest discover -s tests -v   # 必要测试(约 0.1 s)
python demo.py                            # 演示(约 1 s, 真实计算)
```

`demo.py` 使用固定输入, 打印: 一个正常算例的字符布局与最优性状态、
实际触发的输入拒绝(不连通零件/越界禁放格/bool 参数)、
以及实际触发的预算耗尽(与最优/无解状态对比)。

`demo_iteration.py` 演示本次新增的整刀检查(约 0.1 s, 真实计算):
一个可整刀布局的逐刀子板范围与切割树、一个 4x4 风车形(5 件铺满)
无第一刀布局的逐切线阻挡见证、以及预算耗尽与确定不可切的对比。

## 设计取舍

- **位集回溯**: 每个候选放置预计算两个 Python int 位掩码(占用 + Chebyshev
  一格膨胀), 冲突检测为 O(1) 位运算; 膨胀在板边界裁剪(板外不计空间)。
- **等形对称去重**: 同一零件由对称性产生的相同方向只保留代表元
  (未镜像优先, 再按旋转角); 形状且允许方向完全相同的零件归为一组,
  搜索时同组候选下标单调不减以破缺对称。注意: 形状相同但允许旋转集不同
  的零件**不**同组(旋转次数目标不同, 不可互换)。
- **两阶段驱动**: 利用可行性关于高度的单调性 —— 先在整板高度下贪心取
  任一可行解(首叶即停, 同时整板不可行 ⇔ 整体不可行), 再从下界向上逐高度
  完整枚举, 首次完整枚举成功的高度即最优高度, 该高度的枚举同时完成
  旋转次数与字典序的次序优化。
- **MRV + 截断扫描**: 每节点选择可行候选最少的零件组; 扫描带计数截断。
- **预算语义**: 单一计数器贯穿所有阶段; 耗尽即抛内部异常并汇报
  最优已知解与下界, 绝不把未证明结果标记为 `OPTIMAL`/`INFEASIBLE`。

## 已知限制

- 本模块实现的是本题契约的有限子集(网格正交排样 + Chebyshev 一格边距),
  **不宣称兼容任何行业排样标准**(不支持连续坐标、任意多边形、任意角度)。
- 精确回溯求解, 最坏情况指数级; 接近 20×20/12 件上限的紧凑实例可能在
  默认预算(1,000,000 节点)内只能给出 `BUDGET_EXHAUSTED` + 最优已知解,
  这是设计内的诚实报告, 不是错误。调大 `budget` 可继续搜索。
- **误差口径**: 全部输入输出与中间计算均为整数, 无任何浮点结果,
  因此没有数值误差; 任何浮点输入(包括 NaN/Infinity)直接被拒绝。
- 测试中的独立参考实现(`tests/_reference.py`)为逐格穷举, 只适用于
  小规模交叉验证, 不是性能参照。
- 整刀检查只处理**网格线上的轴对齐整刀**(这是本题契约的切割模型),
  不支持斜切、非贯穿刀、或先叠合再切; 判定的是给定固定布局,
  不通过重新排样来"挽救"不可切的布局。
- 整刀检查的独立参考(`tests/_guillotine_ref.py`)为朴素递归枚举,
  同样只用于小规模交叉验证。
