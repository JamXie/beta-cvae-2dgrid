# Beta-CVAE with 2D Grid Representation

Continuous prediction of the 3D deformation and Mises stress fields of thin-walled
structures under impact loading. The method unfolds the closed shell surface into a
planar 2D grid, encodes it with a multi-branch Beta-CVAE, interpolates the latent
space with radial basis functions (RBFI), and decodes arbitrary `(time, angle)`
conditions back to 3D fields.

This repository accompanies Chapter 5 of the author's doctoral thesis
(Chinese: 《薄壁结构冲击响应连续三维大变形预测的几何深度学习方法研究》).
It is a cleaned, runnable reorganization of the chapter's research code — not a
redistribution of the original project.

> **Data and weights are not included.** The finite-element CSV files, converted
> `.npy` grids and trained checkpoints are not part of this repository. Use the
> synthetic generator for a self-contained smoke run, or `convert-csv` for your
> own FE exports.

## Method at a glance

```text
FE CSV  ->  voxelize  ->  3D shell graph  ->  unfold  ->  2D grid  [4, 80, 84]
                                                               |
                                      pad to 96x96, mask border |
                                                               v
                     multi-branch residual encoder -> latent z (16)
                                                               |
                     condition embedding (time, angle)         |
                                                               v
                                   Beta-CVAE training with masked MSE + beta * KL
                                                               |
        RBFI over (step, angle) -> z*  ->  decoder  ->  [4, 96, 96] -> crop
                                                               |
                                          denormalize and compute MAE / RMSE
```

- **2D grid:** channels `dx, dy, dz, Mises`; ring direction 80 nodes, height
  direction 84 layers. Displacements are divided by 300 mm, stress by 200 MPa.
- **Final model:** `M_Rm_CVAE02` — four independent residual encoders, shared
  upsampling head with four independent output branches, condition mode 02
  (time/angle embeddings are sampled as `N(embedding, 1)` during training and
  used as their means at inference).
- **Loss:** masked MSE on the valid window plus a beta-weighted KL term, exactly
  as in the thesis implementation.

See [docs/METHOD.md](docs/METHOD.md) for the formula-to-code mapping.

## Installation

```bash
python -m pip install -r requirements.txt
# optional: editable install
python -m pip install -e .
```

Tested with Python 3.10 and CPU-only PyTorch; CUDA is used automatically when
available. All scripts accept `--device auto|cpu|cuda`.

## Quick start (synthetic data, CPU)

```bash
# 1. generate a small synthetic dataset (npy + a few CSV frames + geometry)
python -m beta_cvae.cli generate-synthetic --out data/synthetic --profile smoke

# 2. train for one epoch (two mini-batches) as a smoke test
python -m beta_cvae.cli train \
  --data data/synthetic/npy --model M_Rm_CVAE02 \
  --beta 1e-6 --epochs 1 --batch-size 2 --max-batches 2 \
  --device cpu --out runs/smoke

# 3. fit RBFI on the training split and evaluate on the validation split
python -m beta_cvae.cli validate-rbfi \
  --checkpoint runs/smoke/checkpoint.pt --data data/synthetic/npy \
  --fit-split train --eval-split val --epsilon 1.8 \
  --device cpu --out runs/smoke/val

# 4. continuous prediction at angle 23 degrees
python -m beta_cvae.cli predict \
  --checkpoint runs/smoke/checkpoint.pt --data data/synthetic/npy \
  --angle 23 --steps 0 50 100 150 200 250 300 350 400 450 500 \
  --geometry data/synthetic/geometry.npz \
  --device cpu --out runs/smoke/pred
```

Expected outputs: `checkpoint.pt`, `loss.png`, `history.csv`, `metrics.csv`,
`summary.json`, `predictions.npz`, channel-comparison PNGs, a prediction montage
and 3D point-cloud PNGs. The smoke run takes seconds on CPU; its error values
are meaningless because the model is trained for two batches only.

## Real finite-element data

Convert a directory of FE CSV frames (columns follow the original exporter:
displacement at positions 2–4, coordinates at 9–11, Mises stress at 12):

```bash
python -m beta_cvae.cli convert-csv \
  --input-dir RAW/00/work0 \
  --reference RAW/00/work0/Degree0_Step0_F-0.0.csv \
  --output-dir data/real/npy \
  --voxel-size 3 \
  --xyz-min -150 -150 0 --xyz-max 150 150 300 \
  --step-stride 5 --plane-vector 0 -1 0 --fast
```

The reference frame determines the voxel occupancy and the node-to-grid mapping
(`geometry.npz`). Later frames reuse that mapping; if a frame has a different
node count, the converter falls back to a full unfold and prints a warning.
See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for details.

## Training and ablation scripts

The unified CLI exposes every model in the registry (`--model`). For batch
experiments use the scripts below; each writes one CSV row per configuration.

```bash
# beta from 1000 down to 1e-7 on the final architecture
python scripts/run_beta_sweep.py --data data/real/npy --out runs/beta_sweep

# CVAE condition modes (01-04) against the final model
python scripts/run_condition_ablation.py --data data/real/npy --out runs/condition

# multi-branch / residual / final architectures
python scripts/run_architecture_ablation.py --data data/real/npy --out runs/architecture
```

Scripts are thin wrappers around `beta_cvae.cli`; add `--epochs`, `--max-batches`
and `--device` as needed for a quick check.

## Repository layout

```text
beta_cvae/
  constants.py          grid shape, padding, normalization, split rules
  cli.py                generate-synthetic / convert-csv / train / validate-rbfi / predict
  data/                 voxelize, unfold, datasets, CSV conversion, synthetic generator
  models/               VAE, CVAE01-04, multi-branch, residual, final models, registry
  training/             masked loss, trainer, checkpoint I/O
  inference/            latent extraction, RBFI, unified decoder
  metrics.py            displacement/stress MAE and RMSE
  visualization.py      loss curve, channel comparison, 3D point cloud (PNG)
scripts/                beta sweep and two ablation runners
tests/                  data, models, training, inference, end-to-end smoke tests
docs/                   method, data format, reproducibility, fixed issues, figures
```

## Tests

```bash
python -m pytest -q
```

The suite runs entirely on CPU without real data: it generates synthetic grids,
checks the unfold round trip, exercises all 19 registered models, verifies the
loss terms and runs the full CLI pipeline on a tiny configuration.

## Historical figures

`docs/figures/` contains a few figures reproduced from the author's thesis for
context (framework, network structure, progressive crushing and global bending
predictions). They are **historical results from the thesis, not outputs of this
repository**. The paper's reported errors (e.g. 3.60 mm displacement and
9.70 MPa stress) cannot be reproduced without the original data and weights.

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
