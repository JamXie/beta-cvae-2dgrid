# 基于 2D Grid 表征的 Beta-CVAE

面向薄壁结构冲击工况，连续预测三维大变形与 Mises 应力场。方法将封闭壳层曲面
沿环向切开展平为规则 2D Grid，用多分支 Beta-CVAE 编码，在潜空间中用径向基函数
（RBFI）做条件插值，再由解码器生成任意 `(时间步, 角度)` 下的三维场。

本仓库对应作者学位论文《薄壁结构冲击响应连续三维大变形预测的几何深度学习方法研究》
第 5 章，是该章研究代码的整理与可运行版本，并非原始工程的完整复制。

> **不含真实数据与训练权重。** 有限元 CSV、转换后的 `.npy` 网格与检查点均不在仓库中。
> 可直接用合成数据生成器完成自包含的冒烟运行，或用 `convert-csv` 接入自己的有限元导出。

## 方法概览

```text
FE CSV（有限元导出）
   -> 体素化并沿环向展开为 2D Grid [4, 80, 84]          beta_cvae/data/
   -> 补零到 96×96，掩码只保留 80×84 有效区域            training/trainer.py
   -> 多分支残差编码器 -> 16 维结构潜变量 z              models/final.py
   -> Beta-CVAE 训练：掩码 MSE + beta × KL               training/losses.py
   -> RBFI 在 (Step, Degree) 上插值潜变量                inference/rbfi.py
   -> 条件解码并裁剪回 [4, 80, 84]                       inference/decode.py
   -> 反归一化，计算位移/应力 MAE、RMSE                  metrics.py
```

潜变量流形负责把高维网格压缩到 16 维；RBFI 在时间-角度条件空间上连续插值，
因此无需为每个时间步或角度单独训练。

- **2D Grid**：通道为 `dx, dy, dz, Mises`；环向 80 个节点，高度 84 层。
  位移除以 300 mm，应力除以 200 MPa。
- **最终模型**：`M_Rm_CVAE02`——四通道独立残差编码器，共享上采样头接四个独立
  输出分支，条件模式 02（训练时按 `N(嵌入, 1)` 采样时间/角度条件，推理时取均值）。
- **损失函数**：有效区域掩码 MSE 加 beta 加权 KL 项，与论文实现保持一致。

公式与代码的逐条对应见 [docs/METHOD.md](docs/METHOD.md)。

## 安装

```bash
python -m pip install -r requirements.txt
# 可选：可编辑安装
python -m pip install -e .
```

在 Python 3.10、纯 CPU 的 PyTorch 环境下测试通过；有 CUDA 时会自动使用。
所有命令均可用 `--device auto|cpu|cuda` 指定设备。

## 快速开始（合成数据，CPU）

```bash
# 1. 生成小型合成数据（npy + 少量 CSV 帧 + 几何文件）
python -m beta_cvae.cli generate-synthetic --out data/synthetic --profile smoke

# 2. 训练 1 个 epoch（仅 2 个 mini-batch），用于冒烟验证
python -m beta_cvae.cli train \
  --data data/synthetic/npy --model M_Rm_CVAE02 \
  --beta 1e-6 --epochs 1 --batch-size 2 --max-batches 2 \
  --device cpu --out runs/smoke

# 3. 在训练集上拟合 RBFI，在验证集上评估
python -m beta_cvae.cli validate-rbfi \
  --checkpoint runs/smoke/checkpoint.pt --data data/synthetic/npy \
  --fit-split train --eval-split val --epsilon 1.8 \
  --device cpu --out runs/smoke/val

# 4. 在 23° 角度上做连续时间步预测
python -m beta_cvae.cli predict \
  --checkpoint runs/smoke/checkpoint.pt --data data/synthetic/npy \
  --angle 23 --steps 0 50 100 150 200 250 300 350 400 450 500 \
  --geometry data/synthetic/geometry.npz \
  --device cpu --out runs/smoke/pred
```

产物包括 `checkpoint.pt`、`loss.png`、`history.csv`、`metrics.csv`、`summary.json`、
`predictions.npz`、通道对比图、时间步拼图与三维点云图。CPU 上数秒即可完成；
由于只训练两个 batch，冒烟运行的误差数值没有物理意义。

## 接入真实有限元数据

按原始导出格式准备 CSV（位移位于第 2–4 列，坐标位于第 9–11 列，Mises 应力位于第 12 列），
然后：

```bash
python -m beta_cvae.cli convert-csv \
  --input-dir RAW/00/work0 \
  --reference RAW/00/work0/Degree0_Step0_F-0.0.csv \
  --output-dir data/real/npy \
  --voxel-size 3 \
  --xyz-min -150 -150 0 --xyz-max 150 150 300 \
  --step-stride 5 --plane-vector 0 -1 0 --fast
```

参考帧决定体素占用与“节点—网格”映射，并保存为 `geometry.npz`；后续帧复用该映射。
若某帧节点数与参考帧不一致，转换器会回退到完整展开并打印警告。
格式细节见 [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)。

## 训练与消融脚本

统一 CLI 通过 `--model` 选择注册表中的任意模型。批量实验请用以下脚本，
每个配置输出一行 CSV：

```bash
# 最终结构上的 beta 扫描（1000 到 1e-7）
python scripts/run_beta_sweep.py --data data/real/npy --out runs/beta_sweep

# CVAE 条件模式（01–04）与最终模型对比
python scripts/run_condition_ablation.py --data data/real/npy --out runs/condition

# 多分支 / 残差 / 最终结构对比
python scripts/run_architecture_ablation.py --data data/real/npy --out runs/architecture
```

脚本本质是 `beta_cvae.cli` 的薄封装；快速检查时可加 `--epochs`、`--max-batches`
和 `--device`。

## 目录结构

```text
beta_cvae/
  constants.py          网格形状、填充、归一化常数、划分规则
  cli.py                generate-synthetic / convert-csv / train / validate-rbfi / predict
  data/                 体素化、展开、数据集、CSV 转换、合成数据生成器
  models/               VAE、CVAE01-04、多分支、残差、最终模型与注册表
  training/             掩码损失、训练循环、检查点读写
  inference/            潜变量提取、RBFI、统一解码入口
  metrics.py            位移/应力 MAE 与 RMSE
  visualization.py      损失曲线、通道对比、三维点云（PNG）
scripts/                beta 扫描与两类消融脚本
tests/                  数据、模型、训练、推理与端到端冒烟测试
docs/                   方法、数据格式、复现说明、修复记录、论文图
```

## 测试

```bash
python -m pytest -q
```

测试全部在 CPU 上运行且不需要真实数据：生成合成网格、检查展开往返一致性、
遍历 19 个注册模型、验证损失项，并用极小配置跑通完整 CLI 流程。

## 论文图说明

`docs/figures/` 收录了作者学位论文中的少量插图，用于方法背景展示
（框架、网络结构、渐进压溃与全局弯曲预测）。这些是**论文的历史结果，
不是本仓库的输出**。缺少原始数据与权重时，无法复现论文报告的误差
（例如位移 3.60 mm、应力 9.70 MPa）。

## 许可证

MIT，详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
