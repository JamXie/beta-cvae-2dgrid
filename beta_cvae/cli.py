"""统一命令行入口。

用法示例：

```bash
python -m beta_cvae.cli generate-synthetic --out data/synthetic --profile smoke
python -m beta_cvae.cli train --data data/synthetic/npy --model M_Rm_CVAE02 \
    --beta 1e-6 --epochs 1 --batch-size 2 --max-batches 2 --device cpu --out runs/smoke
python -m beta_cvae.cli validate-rbfi --checkpoint runs/smoke/checkpoint.pt \
    --data data/synthetic/npy --fit-split train --eval-split val --device cpu --out runs/smoke/val
python -m beta_cvae.cli predict --checkpoint runs/smoke/checkpoint.pt --angle 23 \
    --steps 0 50 100 150 200 250 300 350 400 450 500 \
    --geometry data/synthetic/geometry.npz --device cpu --out runs/smoke/pred
```
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from .constants import (
    DEFAULT_PREDICT_STEPS,
    DEFAULT_TEST_ANGLE,
    LATENT_DIM,
    LATENT_NUM,
    TRAIN_DEGREE_MOD,
    TRAIN_STEP_MOD,
    VAL_PRIME_ANGLES,
)
from .data.convert import convert_csv_directory
from .data.datasets import NpyGridDataset, split_samples
from .data.synthetic import generate_synthetic
from .inference.decode import crop_prediction, decode_from_latent
from .inference.latent import extract_latents
from .inference.rbfi import fit_rbfi, query_rbfi
from .metrics import grid_metrics
from .models.registry import DEFAULT_MODEL, MODEL_NAMES, build_model
from .training.checkpoint import load_checkpoint, save_checkpoint
from .training.trainer import train_model
from .utils import ensure_dir, resolve_device, set_seed
from .visualization import (
    save_3d_prediction,
    save_channel_comparison,
    save_loss_curve,
    save_prediction_montage,
)


def _add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--degree-mod", type=int, default=None, help="训练角度模数（默认取检查点或 5）")
    parser.add_argument("--step-mod", type=int, default=None, help="时间步模数（默认取检查点或 5）")
    parser.add_argument("--test-angle", type=int, default=None, help="测试/预测角度（默认 23）")


def _make_loader(dataset: NpyGridDataset, batch_size: int, shuffle: bool, num_workers: int, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        generator=generator if shuffle else None,
    )


def _data_kwargs(args, config: dict | None = None) -> dict:
    config = config or {}
    degree_mod = args.degree_mod if args.degree_mod is not None else config.get("degree_mod", TRAIN_DEGREE_MOD)
    step_mod = args.step_mod if args.step_mod is not None else config.get("step_mod", TRAIN_STEP_MOD)
    test_angle = args.test_angle if args.test_angle is not None else config.get("test_angle", DEFAULT_TEST_ANGLE)
    return {
        "test_angle": test_angle,
        "val_angles": tuple(VAL_PRIME_ANGLES),
        "train_degree_mod": degree_mod,
        "step_mod": step_mod,
    }


def _load_model(path, device):
    payload = load_checkpoint(path, device=device)
    config = payload.get("config", {})
    model = build_model(
        payload.get("model_name", DEFAULT_MODEL),
        latent_num=config.get("latent_num", LATENT_NUM),
        latent_value=config.get("latent_value", LATENT_DIM),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload


# ================= 子命令实现 =================
def cmd_generate_synthetic(args) -> None:
    meta = generate_synthetic(args.out, profile=args.profile, seed=args.seed)
    print(json.dumps({k: meta[k] for k in ("profile", "counts", "steps", "grid_shape")}, indent=2))


def cmd_convert_csv(args) -> None:
    meta = convert_csv_directory(
        args.input_dir,
        args.reference,
        args.output_dir,
        voxel_size=args.voxel_size,
        xyz_min=tuple(args.xyz_min),
        xyz_max=tuple(args.xyz_max),
        step_stride=args.step_stride,
        degree_stride=args.degree_stride,
        limit=args.limit,
        fast=args.fast,
        plane_vector=tuple(args.plane_vector),
    )
    print(json.dumps({k: meta[k] for k in ("num_frames", "full_unfold_fallbacks", "grid_shape")}, indent=2))


def cmd_train(args) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = ensure_dir(args.out)

    data_kwargs = _data_kwargs(args)
    train_samples = split_samples(args.data, "train", **data_kwargs)
    val_samples = split_samples(args.data, "val", **data_kwargs)

    train_dataset = NpyGridDataset(train_samples)
    val_dataset = NpyGridDataset(val_samples)
    train_loader = _make_loader(train_dataset, args.batch_size, True, args.num_workers, args.seed)
    val_loader = _make_loader(val_dataset, args.batch_size, False, args.num_workers, args.seed)

    model = build_model(
        args.model,
        latent_num=args.latent_num,
        latent_value=args.latent_dim,
    ).to(device)

    print(
        f"model={args.model} train={len(train_dataset)} val={len(val_dataset)} "
        f"beta={args.beta} device={device}"
    )

    history = train_model(
        model,
        train_loader,
        val_loader,
        beta=args.beta,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        max_batches=args.max_batches,
    )

    config = {
        "latent_num": args.latent_num,
        "latent_value": args.latent_dim,
        "degree_mod": data_kwargs["train_degree_mod"],
        "step_mod": data_kwargs["step_mod"],
        "test_angle": data_kwargs["test_angle"],
        "beta": args.beta,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
    }
    checkpoint_path = save_checkpoint(
        out_dir / "checkpoint.pt",
        model,
        model_name=args.model,
        config=config,
        history=history,
    )

    # 损失曲线使用实际记录长度
    save_loss_curve(history, out_dir / "loss.png")

    import csv

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history.keys()))
        writer.writeheader()
        for row in zip(*history.values()):
            writer.writerow(dict(zip(history.keys(), row)))

    print(f"saved checkpoint: {checkpoint_path}")


def cmd_validate_rbfi(args) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = ensure_dir(args.out)

    model, payload = _load_model(args.checkpoint, device)
    data_kwargs = _data_kwargs(args, payload.get("config", {}))

    fit_samples = split_samples(args.data, args.fit_split, **data_kwargs)
    eval_samples = split_samples(args.data, args.eval_split, **data_kwargs)

    fit_loader = _make_loader(NpyGridDataset(fit_samples), args.batch_size, False, args.num_workers, args.seed)
    eval_loader = _make_loader(NpyGridDataset(eval_samples), args.batch_size, False, args.num_workers, args.seed)

    points, values = extract_latents(model, fit_loader, device)
    rbf = fit_rbfi(points, values, epsilon=args.epsilon, smoothing=args.smoothing)
    print(f"fitted RBFI on {len(points)} samples; evaluating {len(eval_samples)} frames")

    predictions = []
    targets = []
    degrees = []
    steps = []
    rows = []

    model.eval()
    for grid, degree, step in eval_loader:
        grid = grid.to(device)
        degree = degree.to(device)
        step = step.to(device)

        latent = query_rbfi(rbf, step.cpu().numpy(), degree.cpu().numpy())
        pred = decode_from_latent(model, latent, step.cpu().numpy(), degree.cpu().numpy(), device)
        pred = crop_prediction(pred).cpu().numpy()
        target = grid.cpu().numpy()

        predictions.append(pred)
        targets.append(target)
        degrees.extend(degree.cpu().numpy().tolist())
        steps.extend(step.cpu().numpy().tolist())

        for i in range(pred.shape[0]):
            metrics = grid_metrics(pred[i], target[i])
            metrics["degree"] = int(degree[i].item())
            metrics["step"] = int(step[i].item())
            rows.append(metrics)

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    summary = grid_metrics(predictions, targets)

    np.savez(
        out_dir / "predictions.npz",
        predictions=predictions,
        targets=targets,
        degrees=np.asarray(degrees, dtype=np.int64),
        steps=np.asarray(steps, dtype=np.int64),
    )

    import csv

    fieldnames = [
        "degree", "step",
        "mae_x_mm", "mae_y_mm", "mae_z_mm",
        "rmse_x_mm", "rmse_y_mm", "rmse_z_mm",
        "euclidean_mae_mm", "euclidean_rmse_mm",
        "stress_mae_mpa", "stress_rmse_mpa",
    ]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    # 为前若干帧保存预测/目标对比图
    for i in range(min(args.num_plots, predictions.shape[0])):
        save_channel_comparison(
            predictions[i],
            targets[i],
            out_dir / f"prediction_{degrees[i]}_{steps[i]}.png",
            title=f"angle={degrees[i]} step={steps[i]}",
        )

    print(json.dumps(summary, indent=2, ensure_ascii=True))


def cmd_predict(args) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = ensure_dir(args.out)

    model, payload = _load_model(args.checkpoint, device)
    data_kwargs = _data_kwargs(args, payload.get("config", {}))

    fit_samples = split_samples(args.data, args.fit_split, **data_kwargs)
    fit_loader = _make_loader(NpyGridDataset(fit_samples), args.batch_size, False, args.num_workers, args.seed)

    points, values = extract_latents(model, fit_loader, device)
    rbf = fit_rbfi(points, values, epsilon=args.epsilon, smoothing=args.smoothing)

    steps = np.asarray(args.steps, dtype=np.int64)
    latent = query_rbfi(rbf, steps, np.full_like(steps, args.angle, dtype=np.float64))

    grids = []
    for start in range(0, len(steps), args.batch_size):
        chunk = slice(start, start + args.batch_size)
        pred = decode_from_latent(model, latent[chunk], steps[chunk], args.angle, device)
        grids.append(crop_prediction(pred).cpu().numpy())
    grids = np.concatenate(grids, axis=0)

    np.savez(
        out_dir / "predictions.npz",
        grids=grids,
        steps=steps,
        angle=np.asarray([args.angle], dtype=np.int64),
        latent=latent,
    )
    save_prediction_montage(grids, steps, out_dir / "prediction_montage.png", angle=args.angle)

    if args.geometry:
        for step, grid in zip(steps, grids):
            save_3d_prediction(
                grid,
                args.geometry,
                out_dir / f"prediction_3d_{int(step)}.png",
                angle=args.angle,
                step=int(step),
            )

    print(f"predicted {len(steps)} frames at angle {args.angle}; saved to {out_dir}")


# ================= 参数解析 =================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Beta-CVAE 2D-grid pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 合成数据
    gen = subparsers.add_parser("generate-synthetic", help="生成合成 npy/CSV/geometry")
    gen.add_argument("--out", default="data/synthetic")
    gen.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    gen.add_argument("--seed", type=int, default=0)
    gen.set_defaults(func=cmd_generate_synthetic)

    # 真实 CSV 转换
    conv = subparsers.add_parser("convert-csv", help="有限元 CSV -> 2D Grid npy")
    conv.add_argument("--input-dir", required=True)
    conv.add_argument("--reference", required=True, help="参考帧 Degree0_Step0_*.csv")
    conv.add_argument("--output-dir", required=True)
    conv.add_argument("--voxel-size", type=float, default=3.0)
    conv.add_argument("--xyz-min", type=float, nargs=3, default=[-150.0, -150.0, 0.0])
    conv.add_argument("--xyz-max", type=float, nargs=3, default=[150.0, 150.0, 300.0])
    conv.add_argument("--step-stride", type=int, default=1)
    conv.add_argument("--degree-stride", type=int, default=1)
    conv.add_argument("--limit", type=int, default=None)
    conv.add_argument("--plane-vector", type=float, nargs=3, default=[0.0, -1.0, 0.0])
    conv.add_argument("--fast", action="store_true", help="复用参考帧 node_positions")
    conv.set_defaults(func=cmd_convert_csv)

    # 训练
    train = subparsers.add_parser("train", help="训练单个模型")
    train.add_argument("--data", required=True, help="npy 所在目录")
    train.add_argument("--model", choices=MODEL_NAMES, default=DEFAULT_MODEL)
    train.add_argument("--beta", type=float, default=1e-6)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    train.add_argument("--latent-num", type=int, default=LATENT_NUM)
    train.add_argument("--max-batches", type=int, default=None)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--out", default="runs/train")
    _add_common_data_args(train)
    train.set_defaults(func=cmd_train)

    # RBFI 验证
    val = subparsers.add_parser("validate-rbfi", help="RBFI 插值 + 解码 + 误差评估")
    val.add_argument("--checkpoint", required=True)
    val.add_argument("--data", required=True)
    val.add_argument("--fit-split", choices=("train", "val", "test"), default="train")
    val.add_argument("--eval-split", choices=("train", "val", "test"), default="val")
    val.add_argument("--epsilon", type=float, default=1.8)
    val.add_argument("--smoothing", type=float, default=0.0)
    val.add_argument("--batch-size", type=int, default=8)
    val.add_argument("--num-workers", type=int, default=0)
    val.add_argument("--num-plots", type=int, default=3)
    val.add_argument("--device", default="auto")
    val.add_argument("--seed", type=int, default=0)
    val.add_argument("--out", default="runs/validate")
    _add_common_data_args(val)
    val.set_defaults(func=cmd_validate_rbfi)

    # 连续预测
    pred = subparsers.add_parser("predict", help="在给定角度与时间步上连续预测")
    pred.add_argument("--checkpoint", required=True)
    pred.add_argument("--data", required=True)
    pred.add_argument("--angle", type=int, default=DEFAULT_TEST_ANGLE)
    pred.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_PREDICT_STEPS))
    pred.add_argument("--fit-split", choices=("train", "val", "test"), default="train")
    pred.add_argument("--geometry", default=None, help="geometry.npz，用于三维图")
    pred.add_argument("--epsilon", type=float, default=1.8)
    pred.add_argument("--smoothing", type=float, default=0.0)
    pred.add_argument("--batch-size", type=int, default=8)
    pred.add_argument("--num-workers", type=int, default=0)
    pred.add_argument("--device", default="auto")
    pred.add_argument("--seed", type=int, default=0)
    pred.add_argument("--out", default="runs/predict")
    _add_common_data_args(pred)
    pred.set_defaults(func=cmd_predict)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
