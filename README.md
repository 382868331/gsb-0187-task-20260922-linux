# nesting.py — 约束二维材料片的无重叠排样

在整数网格板上排布若干 4-连通的格子零件（polyomino），板可含禁放格，
零件可限制旋转角与镜像，求**最小使用高度**排样。纯 Python 标准库实现，
离线运行，无任何第三方依赖。

## 运行环境

- Python **3.14.7**（仅使用 3.14 标准库；不依赖第三方库、网络、账号、
  Docker 或任何外部数据/附件）。
- Linux 原生运行；装好 Python 后即可直接执行下方命令。

## 快速命令

```bash
python -m unittest discover -s tests -v   # 运行全部测试
python demo.py                            # 固定输入演示（正常结果 + 边界）
```

两条命令均为秒级完成；`demo.py` 真实执行搜索（不使用 sleep、
不预录结果），输出字符布局与最优性状态。

## 问题契约（本题精确子集）

- **板**：`width × height` 整数网格，`1 ≤ width, height ≤ 20`，
  坐标 `(x, y)`，`x` 向右增大、`y` 向上增大，底行为 `y = 0`。
  禁放格由 `Board.forbidden` 给出，零件格不得覆盖。
- **零件**：非空、无重复、4-连通（仅上下左右相邻算连通）的格子集合，
  零件数 `1 ≤ n ≤ 12`。输入坐标任意，内部按平移归一化。
- **朝向**：仅支持 0/90/180/270 度旋转（90 度步进，网格严格对齐）；
  变换顺序为**先镜像（x → −x）再逆时针旋转**。每个零件可用
  `allowed_rotations`（非空子集）与 `allow_mirror` 限制可用朝向。
- **边距**：任意两个不同零件的占用格之间 Chebyshev 距离必须 ≥ 2
  （即至少间隔一个空格，含对角方向）。实现上要求新零件的占用格不落入
  已放零件的一格 Chebyshev 膨胀区。膨胀区允许越出板外（**板外不计
  额外空间**）、允许覆盖禁放格；使用高度只按零件实际占用格计，
  即 `max(y) + 1`，顶边外的膨胀不加高。
- **优化目标（严格字典序）**：
  1. 使用高度最小；
  2. 同高度下**旋转次数**最少（仅计 `rotation != 0` 的零件数，
     镜像不计）；
  3. 再相同则布局坐标字典序最小——按零件输入顺序拼接
     `(x, y, code)`，其中 `code = 旋转序号(0..3) * 2 + 镜像位(0/1)`
     （旋转序号 0/1/2/3 分别对应 0/90/180/270 度）。

### 数值与误差口径

全部输入与输出均为**整数**（坐标、尺寸、角度、旋转次数、节点数）。
旋转只支持 90 度步进且作用于整数格点，不存在近似几何；本库
**不产生任何浮点结果，因此没有浮点误差/容差口径**，也不使用任何
宽容差掩盖边界冲突（边距冲突按精确 Chebyshev 距离判定）。

## 公开接口

```python
from nesting import Board, Piece, solve, render, placed_cells

board = Board(width=8, height=6, forbidden=frozenset({(4, 0)}))
piece = Piece(
    cells=((0, 0), (1, 0), (0, 1)),     # 4-连通格子序列
    allowed_rotations=(0, 90, 180, 270),
    allow_mirror=False,
)
result = solve(board, [piece], budget=200_000)
print(render(result, board, [piece]))
```

- `Board(width, height, forbidden=frozenset())`
- `Piece(cells, allowed_rotations=(0,90,180,270), allow_mirror=False)`
- `solve(board, pieces, budget=DEFAULT_BUDGET) -> NestingResult`
- `Placement(piece_index, x, y, rotation, mirrored)`
- `NestingResult` 字段：

  | 字段 | 含义 |
  |---|---|
  | `status` | `"optimal"` / `"infeasible"` / `"budget_exhausted"` |
  | `height` | 最优（或最优已知）布局的使用高度；无已知布局为 `None` |
  | `rotation_count` | 该布局旋转次数；无已知布局为 `None` |
  | `placements` | 按零件输入顺序排列的 `Placement` 元组；无布局为 `None` |
  | `lower_bound_height` | **已证明**的使用高度下界；`infeasible` 时为板高 +1 |
  | `nodes`, `budget` | 实际/上限搜索节点数（回溯节点，各优化阶段共享预算） |
  | `message` | 人读说明 |

- `placed_cells(piece, placement)`：返回放置后的绝对格元组。
- `render(result, board=None, pieces=None)`：字符画（顶行在上），
  零件标签 `A..L`，禁放格 `#`，空格 `.`；也可 `result.render()`
  （不带几何信息时只输出状态行）。
- 常量：`MAX_BOARD_DIM=20`、`MAX_PIECES=12`、
  `DEFAULT_BUDGET=200_000`。

### 状态语义（预算耗尽 ≠ 证明无解）

- `optimal`：三级目标均**已证明**最优；`height == lower_bound_height`。
- `infeasible`：已穷举证明板高内不存在任何可行排样；
  `lower_bound_height = height + 1`，`placements is None`。
- `budget_exhausted`：搜索预算在证明完成前耗尽。此时：
  - 若已找到任何可行布局，`height` 是**最优已知**（附带合法布局与
    其旋转次数），`lower_bound_height` 是**已证下界**，
    `lower_bound_height ≤ 真实最优 ≤ height`；状态绝不标成 optimal；
  - 若连板全高可行性都未及判定，则 `height is None`，下界仅为
    几何下界（面积下界与逐零件最小包围盒高度的最大值）。
  报告的已知布局还会做一次纯几何“下沉”压缩（不消耗搜索预算），
  以收紧最优已知高度；压缩不改变其“未证最优”的性质。

`solve` 是纯函数：不修改入参；校验在任何搜索前完成，失败时
不存在部分变更。

## 错误语义

- 所有非法输入抛出 `nesting.NestingError`（继承 `ValueError`），
  错误信息以可定位路径开头，如 `[pieces[2].cells]`、`[Board.width]`、
  `[budget]`；不静默纠正、不猜测默认值。
- 整数参数**拒绝 `bool`**（Python 中 `True/False` 是 `int` 子类，
  本库显式拒绝）；坐标、尺寸、角度、预算均要求严格 `int`。
- 数值输入拒绝 `NaN` 与 `±Infinity`（传入 float 本身即被拒绝）。
- 零件：空集合、重复格、非 4-连通、非法旋转角（仅 0/90/180/270）、
  旋转集合为空或重复、`allow_mirror` 非严格 `bool` 均拒绝。
- 板：尺寸越界（须在 1..20）、禁放格越板均拒绝。
- 求解：零件数越界（须在 1..12）、元素类型不符、`budget < 1` 均拒绝。

## 算法与设计取舍

1. **位集表示**：板面状态用一个 Python 大整数位图表示
  （位索引 `y * width + x`）；零件占用、禁放、Chebyshev 膨胀均为
  整数位运算（移位时用列掩码防止跨行回绕，结果裁剪在板内）。
2. **完备回溯 + MRV**：零件身份固定，逐件枚举全部合法放置；每个
   节点对每个等形组只考虑其剩余最小下标代表，并用“当前可行候选
   最少的组代表”（MRV）分支，候选列表随放置单调过滤。搜索完备：
   任何可行排样都会被找到，证尽则为无解。
3. **等形对称去重**：形状与朝向策略完全相同的零件归为一组；组内
   只按“零件下标升序、放置键 `(x, y, code)` 严格升序”的规范顺序
   落位（任意解按键排序重贴标签后唯一），消除相同零件互换造成的
   重复搜索；零件自身的对称朝向（如正方形旋转）在变体枚举时按
   归一化形状去重。
4. **2×2 块容量必要界**：对齐的 2×2 块（板边缘为 1 宽/高）内任意
   两格 Chebyshev 距离 ≤1，故每个块至多含一个零件的格子；将自由
   格位图折叠到块上统计非空块数，小于剩余零件数即剪枝。
5. **单调性二分**：可行性关于 strip 高度、关于旋转预算都单调，
   因此“最小高度”与“最小旋转次数”均用二分而非线性扫描
   （板高 ≤20 时约 5 次查询），并在过程中维护已证下界与最优已知。
6. **显式搜索预算**：以回溯节点计数（`nodes ≤ budget` 严格成立），
   预算在所有阶段间共享。预算只影响证明程度，绝不影响正确性：
   耗尽时同时报告最优已知与已证下界，并与“已证无解”明确区分。
7. **字典序平局**：在前两级目标证明完成后，逐零件、按 `(x,y,code)`
   枚举其放置，以“前缀可扩展为可行解”为判定，逐位固定最小键。

## 测试

```bash
python -m unittest discover -s tests -v
```

- `tests/reference.py`：**独立**全枚举参考实现，几何逻辑与主程序
  分开重写，**不调用被测核心**得到期望值；
- `tests/helpers.py`：独立布局校验器（重算朝向、逐对检查精确
  Chebyshev 边距、禁放格、板界与使用高度）；
- `tests/test_crosscheck.py`：固定案例（含孔零件、重复形、仅旋转
  可放、仅镜像可放、不可行）+ 40 个确定性随机小板案例，逐项比对
  状态/高度/旋转次数/字典序键；
- `tests/test_solver.py`：边距规则、朝向策略、孔与重复形、三种
  状态语义、预算扫描下已知解始终合法且 `nodes ≤ budget` 等；
- `tests/test_validation.py`：非法输入、NaN/Infinity、bool、可定位
  错误信息、入参不被修改。

## 已知限制（本题子集，不宣称行业标准兼容）

- 仅支持整数网格上轴对齐 90 度步进旋转与单轴镜像；不支持任意角度、
  自由曲线轮廓或连续坐标排样。
- 连通性固定为 4-连通；边距固定为一格 Chebyshev（8 邻域）膨胀。
- 优化目标固定为“高度 → 旋转次数 → 布局字典序”；不优化宽度/面积
  利用率（宽度为固定输入）。
- 最坏情况是指数级组合问题：逼近上限且排布紧凑的不可行/临界实例
  在默认预算内可能无法证尽，此时结果为 `budget_exhausted`（带
  最优已知与已证下界），而非伪造最优；可调大 `budget` 重试。
- 预算按回溯节点计，不含一次性候选生成与几何压缩的耗时；这些
  预处理在本题规模（≤20×20、≤12 件）下开销很小。
