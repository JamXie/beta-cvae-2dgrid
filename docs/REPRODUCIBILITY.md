# 可复现性说明

## 1. 测试环境

本仓库开发与验证所用环境：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.7 |
| PyTorch | 1.13.1+cpu |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| pandas | 1.5.3 |
| Matplotlib | 3.7.0 |
| OS | Windows 10（CPU） |

`requirements.txt` 给出的是下限，较新版本同样可用。PyTorch 1.13 的
`torch.load(weight_only=True)` 不支持嵌套字典，因此检查点把配置与历史
序列化为 JSON 字符串；在 2.x 上加载路径不变。

## 2. 固定随机性

- 所有 CLI 命令都有 `--seed`（默认 0），同时固定 Python、NumPy、PyTorch；
- 合成数据生成固定 `seed=0`；
- 训练 DataLoader 的 shuffle 使用带种子的 `torch.Generator`；
- 条件模式 02 在训练时采样 `N(embedding, 1)`，推理时使用嵌入均值，
  因此连续预测不引入随机噪声。

完全逐位复现仍可能受 BLAS 线程调度影响；小规模测试中已通过
`set_thread_env()` 限制线程数。

## 3. 冒烟流程

```bash
python -m pip install -r requirements.txt
python -m pytest -q

python -m beta_cvae.cli generate-synthetic --out data/synthetic --profile smoke
python -m beta_cvae.cli train \
  --data data/synthetic/npy --model M_Rm_CVAE02 \
  --beta 1e-6 --epochs 1 --batch-size 2 --max-batches 2 \
  --device cpu --out runs/smoke
python -m beta_cvae.cli validate-rbfi \
  --checkpoint runs/smoke/checkpoint.pt --data data/synthetic/npy \
  --fit-split train --eval-split val --epsilon 1.8 \
  --device cpu --num-plots 2 --out runs/smoke/val
python -m beta_cvae.cli predict \
  --checkpoint runs/smoke/checkpoint.pt --data data/synthetic/npy \
  --angle 23 --steps 0 50 100 150 200 250 300 350 400 450 500 \
  --geometry data/synthetic/geometry.npz \
  --device cpu --out runs/smoke/pred
```

预期：

- `pytest` 全部通过（当前 43 项，约 10 秒）；
- 训练在数秒内完成，打印每个 epoch 的 train/val 损失；
- 验证输出 `metrics.csv`、`summary.json`、`predictions.npz` 与对比图；
- 预测输出 11 帧的 `predictions.npz`、拼图与 3D 点云 PNG。

**冒烟误差没有物理意义**：模型只训练了两个 batch，`summary.json` 中的
毫米/兆帕数值显著偏大是正常现象。

## 4. 真实数据流程

准备原始 FE CSV 后：

```bash
python -m beta_cvae.cli convert-csv \
  --input-dir RAW/00/work0 \
  --reference RAW/00/work0/Degree0_Step0_F-0.0.csv \
  --output-dir data/real/npy \
  --voxel-size 3 \
  --xyz-min -150 -150 0 --xyz-max 150 150 300 \
  --step-stride 5 --plane-vector 0 -1 0 --fast

python -m beta_cvae.cli train \
  --data data/real/npy --model M_Rm_CVAE02 \
  --beta 1e-6 --epochs 100 --batch-size 8 \
  --device cuda --out runs/final

python -m beta_cvae.cli validate-rbfi \
  --checkpoint runs/final/checkpoint.pt --data data/real/npy \
  --fit-split train --eval-split val --epsilon 1.8 \
  --device cuda --out runs/final/val
```

转换结果应满足：

```text
npy shape          == (4, 80, 84)
full_unfold_fallbacks == 0        （在节点拓扑一致时）
meta.json 中的 grid_shape == [4, 80, 84]
```

若出现形状警告或大量回退，优先检查 `--voxel-size`、`--xyz-min/--xyz-max`
与参考帧选择。

## 5. 对比实验

```bash
python scripts/run_beta_sweep.py --data data/real/npy \
  --out runs/beta_sweep --epochs 100 --device cuda

python scripts/run_condition_ablation.py --data data/real/npy \
  --out runs/condition --epochs 100 --device cuda

python scripts/run_architecture_ablation.py --data data/real/npy \
  --out runs/architecture --epochs 100 --device cuda
```

每个配置的检查点位于 `<out>/<配置名>/train/`，验证结果位于
`<out>/<配置名>/val/`；汇总表为对应目录下的 CSV。

## 6. 不可复现的部分

真实有限元 CSV、转换后的网格与训练检查点均未随仓库分发。因此：

- 无法直接复现论文报告的 3.60 mm 位移误差与 9.70 MPa 应力误差；
- `docs/figures/` 中的图片是论文历史结果，不是本仓库运行输出；
- 合成数据只验证代码路径，其数据分布与真实冲击响应无关。

若需对照论文数值，请使用作者单位的原始数据与训练权重，并保持与论文一致的
`beta=1e-6`、训练轮数、划分与随机种子。
