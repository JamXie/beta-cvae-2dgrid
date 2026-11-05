# 数据格式

## 1. 有限元 CSV

`convert-csv` 读取原始导出 CSV，列位置沿用原工程约定（0 起始索引）：

| 列 | 内容 | 代码 |
| --- | --- | --- |
| 0 | 行索引 | 忽略 |
| 1 | 节点编号 | 仅用于剔除空行（遇到空值即停止读取） |
| 2–4 | 位移 `U1, U2, U3` | `data.iloc[:, 2:5]` |
| 5–8 | 转动分量等 | 忽略 |
| 9–11 | 节点坐标 `X, Y, Z` | `data.iloc[:, 9:12]` |
| 12 | Mises 应力 | `data.iloc[:, 12:13]` |

参考帧必须是 `Degree0_Step0_*.csv`。参考帧提供两样东西：

1. **体素占用**：哪些体素被结构占据（后续帧保持不变）；
2. **展开映射**：每个 3D 节点落在 2D Grid 的哪个位置。

转换流程（`beta_cvae/data/voxelize.py`、`unfold.py`、`convert.py`）：

```text
参考帧 + 当前帧 CSV
   -> xy 平面中心平移（复制后平移，不修改输入）
   -> 按 voxel_size 与 xyz_min/xyz_max 体素化            [D, H, W, 5]
   -> 通道前置、位移 /300、应力 /200                      [5, D, H, W]
   -> 节点特征 [x, y, z, dx, dy, dz, Mises] + 六邻域边    [N, 7], [E, 2]
   -> 环向展开                                           [4, 80, 84]
```

体素通道含义：

| 通道 | 含义 |
| --- | --- |
| 0 | 占位（参考帧节点落入该体素则为 1） |
| 1–3 | 体素内节点的距离加权平均位移 |
| 4 | 体素内节点的距离加权平均 Mises 应力 |

常用参数：

```bash
--voxel-size 3
--xyz-min -150 -150 0 --xyz-max 150 150 300
--plane-vector 0 -1 0
--step-stride 5 --degree-stride 1 --limit 100
--fast
```

`--fast` 复用参考帧的 `node_positions`；若某帧节点数与参考帧不一致，
会打印 `[warning] ... fell back to full unfold` 并回退到完整展开。

## 2. npy 网格

文件名：

```text
Degree{D}_Step{S}_F{F}.npy
```

例如 `Degree23_Step250_F-0.123.npy`。解析正则见
`beta_cvae/constants.py::FILE_PATTERN`。`F` 只用于文件名，不参与训练。

数组形状：

```text
(4, 80, 84)          # [通道, 环向节点, 高度层]
dtype = float32
```

- 通道 0/1/2：x/y/z 位移，已除以 300；
- 通道 3：Mises 应力，已除以 200；
- 数值大致落在 `[-1, 1]`。

轴顺序由论文实现的 padding 与 mask 反推确认：`F.pad(G, (6,6,8,8))`
把 84 宽的轴补到 96、80 高的轴补到 96，mask 为 `8:88`（80）与 `6:90`（84），
因此第 1 空间轴为环向 80，第 2 空间轴为高度 84。

## 3. geometry.npz

`convert-csv` 与 `generate-synthetic` 都会写出该文件，供三维可视化使用：

| 键 | 形状 | 含义 |
| --- | --- | --- |
| `node_positions` | `[N, 3]` | `[原始节点索引, 环向位置, 高度层]` |
| `edge_index` | `[E, 2]` | 3D 图的无向边（环向相邻 + 层间相邻） |
| `reference_nodes` | `[N, 3]` | 参考帧节点坐标 `(x, y, z)` |
| `grid_index` | `[80, 84]` | 2D 网格位置到 3D 节点索引的映射，空位为 -1 |

`predict --geometry <path>` 用 `reference_nodes + 预测位移 × 300` 绘制三维点云；
不提供 `--geometry` 时只输出 2D 通道图与时间步拼图。

## 4. 数据划分

`beta_cvae/data/datasets.py`：

| 划分 | 角度 | 时间步 |
| --- | --- | --- |
| train | `angle % 5 == 0`（0, 5, …, 45） | `step % 5 == 0` |
| val | 质数 `7, 11, 13, 17, 19, 29, 31, 37, 41, 43` | `step % 5 == 0` |
| test / predict | `23` | `step % 5 == 0` |

注意验证集已移除 23：原工程把 23 同时用于验证和测试，会造成信息泄漏。
`check_split_disjoint` 会在构造数据集时断言三个集合互斥。

`predict` 的 RBFI 默认在 `train` 划分上拟合，因此 `23°` 对插值而言是
真正的外推角度。

## 5. 合成数据

```bash
python -m beta_cvae.cli generate-synthetic --out data/synthetic --profile smoke
python -m beta_cvae.cli generate-synthetic --out data/synthetic --profile full
```

输出目录：

```text
data/synthetic/
├── npy/Degree{D}_Step{S}_F{F}.npy    # [4, 80, 84] float32
├── csv/work0/                        # 4 帧有限元 CSV，用于验证 convert-csv
│   └── Degree0_Step0_F-0.0.csv       # 参考帧
├── geometry.npz
└── meta.json
```

- `smoke`：时间步 `0, 250, 500`，训练角度 `0, 5, …, 45`，验证角度 10 个质数，
  测试角度 23，共 63 帧；
- `full`：时间步 `0, 50, …, 500`，其余划分相同。

合成几何是矩形截面薄壁管：截面周长 240（80 个节点，间距 3 mm）、
84 层（层间距 3 mm），因此按 3 mm 体素转换后恰好得到 `[4, 80, 84]`。
通道值由角度与时间步的解析式生成，固定 `seed=0`，仅用于流程验证，
不代表论文数据分布。
