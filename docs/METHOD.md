# 方法与代码对应

本文档把论文第 5 章的公式与符号映射到仓库中的实现位置。数学形式与损失
写法保持原工程不变，差异仅在工程组织、推理调用方式与数据泄漏修复，
最后一节逐条列出。

## 1. 2D Grid 表征

单帧数据记为

```text
G ∈ R^(4 × 80 × 84)
```

- 通道 0–2：x / y / z 方向位移；
- 通道 3：Mises 应力；
- 第 1 个空间轴为环向节点（80），第 2 个空间轴为高度层（84）。

归一化（`beta_cvae/constants.py`）：

```text
G[c] = raw[c] / 300.0    (c = 0,1,2, 位移, mm)
G[3]  = raw[3]  / 200.0  (应力, MPa)
t     = step / 500.0
a     = angle / 45.0
```

**展开映射**（`beta_cvae/data/unfold.py::unfold_3d_grid_to_2d`）：对每一层
（固定 z）沿环向连通路径切开展平，把路径上第 `i` 个节点的四维特征写入
`G[:, i, layer]`；同时记录 `node_positions = [原始节点索引, i, layer]`。
后续帧用 `fast_unfold_3d_grid_to_2d` 复用该映射，避免重复搜索路径。
逆映射 `_2d_grid_to_3d` 把网格值写回节点，用于测试与可视化。

**网络输入**（`beta_cvae/training/trainer.py`）：

```text
F.pad(G, (6, 6, 8, 8)) -> [B, 4, 96, 96]
mask = 1 on [:, :, 8:88, 6:90], 0 elsewhere
```

裁剪对应 `beta_cvae/inference/decode.py::crop_prediction`，满足
`crop(pad(G)) == G`。

## 2. 基本结构

**残差块**（`beta_cvae/models/basic.py::ResidualBlock`）：

```text
main:  Conv/ConvTranspose(k=3) -> BN -> LeakyReLU(0.2) -> Conv(k=3) -> BN
skip:  恒等映射；通道数或分辨率变化时用 1×1 卷积/转置卷积对齐
out:   LeakyReLU(main + skip)
```

编码器整体为 `32 → 64 → 128 → 256` 四级下采样，再接 `AdaptiveAvgPool2d(1)`；
96×96 经四次下采样得到 6×6，因此默认 `latent_Num = 6`、`latent_value = 16`。

**解码器**（`beta_cvae/models/final.py`）：

```text
shared_decoder: Linear(latent + 2) -> 256×36 -> ReLU -> Unflatten(256,6,6)
                -> ConvTranspose(256→128, k=4, s=2, p=1) -> ReLU
branch:         ConvTranspose(128→64) -> ReLU
                -> ConvTranspose(64→32)  -> ReLU
                -> ConvTranspose(32→1)   -> Tanh
```

三个位移分支与一个应力分支各自独立输出单通道，沿通道维拼接为 `[B,4,96,96]`。

## 3. 条件模式 CVAE01–CVAE04

`beta_cvae/models/basic.py` 与 `final.py` 中的同名类一一对应。
记条件嵌入为

```text
e_t = time_embed(t),    e_a = angle_embed(a)
```

| 模式 | 编码器是否接收条件 | 解码器条件输入 | 训练时的条件潜变量 |
| --- | --- | --- | --- |
| CVAE01 | 是：`[encoder(x), e_t, e_a]` 接 FC | 无（条件已进入 z） | — |
| CVAE02 | 否 | `[z, z_t, z_a]` | `z_t = e_t + ε_t`, `z_a = e_a + ε_a`, `ε ~ N(0,1)` |
| CVAE03 | 否 | `[z, e_t, e_a]` | 直接使用嵌入 |
| CVAE04 | 是 | `[z, e_t, e_a]` | 直接使用嵌入 |

最终模型 `M_Rm_CVAE02` 属于 CVAE02 模式：训练时对条件嵌入加高斯噪声，
使条件潜变量成为随机变量；推理时使用其均值，保证连续预测可重复。

## 4. 网络变体命名

| 论文表述 | 代码类名 | 说明 |
| --- | --- | --- |
| VAE | `VAE` | 单通道共享卷积的基础 VAE |
| M-VAE01/02/03 | `MultiBranchVAE01/02/03` | 多通道独立卷积的三个版本 |
| R-VAE01/02/03 | `ResidualVAE01/02/03` | 残差连接的三个版本 |
| CVAE01–04 | `CVAE01–04` | 四种条件注入方式 |
| `M_R_CVAE01–04` | `M_R_CVAE01–04` | 共享残差编码器 + 多分支解码器 |
| `M_Rm_CVAE01–04` | `M_Rm_CVAE01–04` | 四通道独立残差编码器 + 多分支解码器 |
| **Beta-CVAE（最终）** | **`M_Rm_CVAE02`** | 论文第 5 章最终结构 |

`M_R_*` 与 `M_Rm_*` 的唯一区别是编码器：前者用同一卷积骨干处理四通道输入，
后者为每个通道各用一个单通道残差骨干，再把 4×256 维特征拼接后接
`fc_mu / fc_logvar`。

模型注册表（`beta_cvae/models/registry.py`）为每个模型声明
`decode_style` 与 `encode_takes_condition`，推理代码据此分派，
不依赖对 `decode` 形参个数的猜测：

- `"z"`：`decode(z)`；
- `"z_t_a"`：`decode(z, t, a)`；
- `"z_zt_za"`：先取 `time_embed(t)`、`angle_embed(a)` 的均值，再
  `decode(z, z_t, z_a)`。

## 5. 损失函数

`beta_cvae/training/losses.py`，与原工程完全一致：

```text
L_mse = Σ (pred - target)^2 * mask / Σ mask
L_kl  = beta * 0.5 * Σ (mu^2 + logvar^2 - log(1e-8 + logvar^2) - 1)
                    / (batch_size * latent_dim)
L     = L_mse + L_kl
```

注意 KL 项按论文实现直接使用 `logvar`（而非标准 VAE 中的对数方差），
并保留 `1e-8` 的数值平滑。仓库未将其“修正”为标准 ELBO，以保证与论文
实验一致。

训练循环（`beta_cvae/training/trainer.py`）每步：

```text
pad -> mask -> forward -> L_mse + L_kl -> backward -> Adam.step()
```

固定超参数为 `Adam(lr=1e-4, weight_decay=1e-5)`；论文最优
`beta = 1e-6`。

## 6. RBFI 连续预测

`beta_cvae/inference/latent.py` 与 `rbfi.py`：

1. 对训练划分的每一帧 `x` 取编码器均值 `mu(x)`，并按 `(step, degree)`
   记录采样点；
2. 拟合 `RBFInterpolator(points, mu, kernel="multiquadric", epsilon=1.8)`；
3. 查询目标 `(step*, degree*)` 得到结构潜变量 `z*`；
4. 按注册表的 `decode_style` 解码，裁剪并反归一化。

训练点是原始数值 `(step, degree)`，不做归一化；归一化只发生在条件嵌入前。

## 7. 评价指标

`beta_cvae/metrics.py` 先把预测与目标按通道乘回 300 / 200，再计算：

```text
MAE_axis  = mean |pred_axis - true_axis|            (mm)
RMSE_axis = sqrt(mean (pred_axis - true_axis)^2)    (mm)
E_mae     = mean ||pred_xyz - true_xyz||_2           (mm)
E_rmse    = sqrt(mean (||pred_xyz - true_xyz||_2)^2) (mm)
MAE_σ     = mean |pred_σ - true_σ|                  (MPa)
RMSE_σ    = sqrt(mean (pred_σ - true_σ)^2)          (MPa)
```

## 8. 与原始实现的差异

| 项目 | 原工程 | 本仓库 | 原因 |
| --- | --- | --- | --- |
| 测试角度 | `Degree=5`，同时出现在训练集 | `Degree=23`，从验证集质数中移除 23 | 消除信息泄漏 |
| 条件采样 | 部分推理脚本直接 `decode(z)`，与 `M_Rm_CVAE02` 签名不符 | 统一 `decode_from_latent`，按 `decode_style` 分派；评估用条件均值 | 修复阻塞性错误，保证可重复 |
| 检查点 | `torch.load` 未指定 `map_location` | `load_checkpoint(path, device)`，优先 `weights_only=True` | 安全与跨设备加载 |
| 元信息 | 嵌套字典 pickle | JSON 字符串字段 | 兼容旧版 PyTorch 的 weights-only |
| 测试集划分 | 与训练集重叠 | 构造数据集时断言 train/val/test 角度互斥 | 防止泄漏 |
| 坐标平移 | 原地修改输入数组 | 先复制再平移 | 避免参考坐标被重复平移 |
| 可视化 | 导出 PPTX | 直接保存 PNG + CSV/JSON | 去除 `python-pptx` 依赖 |
| 训练缓存 | 每步无条件 `empty_cache()` | 仅 CUDA 设备调用 | CPU 训练无意义 |
| 路径 | 硬编码 `H:\`、`../data` | `argparse` + `pathlib` | 可移植 |
