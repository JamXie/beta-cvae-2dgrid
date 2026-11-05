"""实验脚本共用：单次“训练 + RBFI 验证”与结果表写入。

三个对比脚本（Beta 扫描、条件模式消融、结构消融）都通过这里复用
`beta_cvae.cli` 的子命令，避免复制训练/验证逻辑。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# 允许从任意工作目录直接运行脚本
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from beta_cvae.cli import main as cli_main  # noqa: E402

# 验证阶段输出的主要误差指标
METRIC_KEYS: tuple[str, ...] = (
    "mae_x_mm",
    "mae_y_mm",
    "mae_z_mm",
    "rmse_x_mm",
    "rmse_y_mm",
    "rmse_z_mm",
    "euclidean_mae_mm",
    "euclidean_rmse_mm",
    "stress_mae_mpa",
    "stress_rmse_mpa",
)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """三个脚本共用的数据、训练与验证参数。"""
    parser.add_argument("--data", required=True, help="npy 数据目录")
    parser.add_argument("--out", default="runs/experiments", help="实验输出根目录")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-batches", type=int, default=None, help="调试用，限制每 epoch 批数")
    parser.add_argument("--fit-split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--eval-split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--epsilon", type=float, default=1.8, help="RBFI 的 multiquadric epsilon")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)


def run_experiment(
    data,
    out_dir,
    *,
    model: str,
    beta: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_batches: int | None,
    fit_split: str,
    eval_split: str,
    epsilon: float,
    device: str,
    seed: int,
) -> dict:
    """训练一个模型，拟合 RBFI 并在评估集上计算误差，返回汇总字典。"""
    out_dir = Path(out_dir)
    train_out = out_dir / "train"
    val_out = out_dir / "val"

    train_argv = [
        "train",
        "--data", str(data),
        "--model", model,
        "--beta", str(beta),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--weight-decay", str(weight_decay),
        "--device", device,
        "--seed", str(seed),
        "--out", str(train_out),
    ]
    if max_batches is not None:
        train_argv += ["--max-batches", str(max_batches)]
    cli_main(train_argv)

    val_argv = [
        "validate-rbfi",
        "--checkpoint", str(train_out / "checkpoint.pt"),
        "--data", str(data),
        "--fit-split", fit_split,
        "--eval-split", eval_split,
        "--epsilon", str(epsilon),
        "--batch-size", str(batch_size),
        "--num-plots", "0",
        "--device", device,
        "--seed", str(seed),
        "--out", str(val_out),
    ]
    cli_main(val_argv)

    return json.loads((val_out / "summary.json").read_text(encoding="utf-8"))


def write_summary_csv(path, rows: list[dict], fieldnames: list[str]) -> Path:
    """把逐组实验指标写成 CSV，便于后续作图。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_table(rows: list[dict], fieldnames: list[str]) -> None:
    """在控制台打印紧凑的结果表（仅 ASCII 表头）。"""
    print(" | ".join(fieldnames))
    for row in rows:
        print(" | ".join(f"{row[key]:.6g}" if isinstance(row[key], float) else str(row[key]) for key in fieldnames))
