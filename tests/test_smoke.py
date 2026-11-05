"""端到端冒烟测试：合成数据 -> 训练 -> RBFI 验证 -> 连续预测 -> CSV 转换。

全部在 CPU 上、使用极小 epoch/批数运行，只验证流程与产物，不代表精度。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from beta_cvae.cli import main as cli_main
from beta_cvae.constants import GRID_SHAPE
from beta_cvae.training.checkpoint import load_checkpoint


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """生成合成数据并跑完整训练/验证/预测流程，返回目录路径。"""
    root = tmp_path_factory.mktemp("smoke")
    data_dir = root / "synthetic"
    runs_dir = root / "runs"

    cli_main(["generate-synthetic", "--out", str(data_dir), "--profile", "smoke"])
    cli_main(
        [
            "train",
            "--data", str(data_dir / "npy"),
            "--model", "M_Rm_CVAE02",
            "--beta", "1e-6",
            "--epochs", "1",
            "--batch-size", "2",
            "--max-batches", "1",
            "--device", "cpu",
            "--out", str(runs_dir / "train"),
        ]
    )
    cli_main(
        [
            "validate-rbfi",
            "--checkpoint", str(runs_dir / "train" / "checkpoint.pt"),
            "--data", str(data_dir / "npy"),
            "--fit-split", "train",
            "--eval-split", "val",
            "--num-plots", "0",
            "--device", "cpu",
            "--out", str(runs_dir / "val"),
        ]
    )
    cli_main(
        [
            "predict",
            "--checkpoint", str(runs_dir / "train" / "checkpoint.pt"),
            "--data", str(data_dir / "npy"),
            "--angle", "23",
            "--steps", "0", "250", "500",
            "--geometry", str(data_dir / "geometry.npz"),
            "--device", "cpu",
            "--out", str(runs_dir / "pred"),
        ]
    )
    return data_dir, runs_dir


def test_synthetic_files_and_shape(smoke_run):
    """合成数据包含三类划分、正确形状与几何文件。"""
    data_dir, _ = smoke_run
    npy_files = sorted((data_dir / "npy").glob("*.npy"))
    assert npy_files

    grid = np.load(npy_files[0], allow_pickle=False)
    assert grid.shape == GRID_SHAPE
    assert grid.dtype == np.float32
    assert np.isfinite(grid).all()

    geometry = np.load(data_dir / "geometry.npz", allow_pickle=False)
    assert geometry["grid_index"].shape == GRID_SHAPE[1:]
    assert geometry["reference_nodes"].shape[1] == 3

    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["grid_shape"] == list(GRID_SHAPE)


def test_training_artifacts_and_checkpoint(smoke_run):
    """训练产生检查点、损失曲线与历史，检查点可安全加载。"""
    _, runs_dir = smoke_run
    train_dir = runs_dir / "train"
    assert (train_dir / "checkpoint.pt").exists()
    assert (train_dir / "loss.png").exists()
    assert (train_dir / "history.csv").exists()

    payload = load_checkpoint(train_dir / "checkpoint.pt", device="cpu")
    assert payload["model_name"] == "M_Rm_CVAE02"
    assert payload["config"]["test_angle"] == 23
    assert payload["history"]["train_loss"]


def test_validation_and_prediction_artifacts(smoke_run):
    """验证与预测产出有限指标、预测数组与图像。"""
    _, runs_dir = smoke_run
    val_dir = runs_dir / "val"
    pred_dir = runs_dir / "pred"

    summary = json.loads((val_dir / "summary.json").read_text(encoding="utf-8"))
    assert np.isfinite(summary["euclidean_mae_mm"])
    assert (val_dir / "metrics.csv").exists()
    predictions = np.load(val_dir / "predictions.npz", allow_pickle=False)
    assert predictions["predictions"].shape[1:] == GRID_SHAPE

    pred = np.load(pred_dir / "predictions.npz", allow_pickle=False)
    assert pred["grids"].shape == (3, *GRID_SHAPE)
    assert (pred_dir / "prediction_montage.png").exists()
    assert (pred_dir / "prediction_3d_0.png").exists()


def test_convert_csv_outputs_expected_shape(smoke_run):
    """真实 CSV 转换路径：快速展开得到 `[4,80,84]`。"""
    data_dir, _ = smoke_run
    csv_dir = data_dir / "csv" / "work0"
    out_dir = data_dir / "converted"

    cli_main(
        [
            "convert-csv",
            "--input-dir", str(csv_dir),
            "--reference", str(csv_dir / "Degree0_Step0_F-0.0.csv"),
            "--output-dir", str(out_dir),
            "--voxel-size", "3",
            "--xyz-min", "-60", "-60", "0",
            "--xyz-max", "60", "60", "252",
            "--plane-vector", "0", "-1", "0",
            "--fast",
            "--limit", "2",
        ]
    )

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["num_frames"] == 2
    assert meta["full_unfold_fallbacks"] == 0

    for name in meta["files"]:
        grid = np.load(out_dir / name, allow_pickle=False)
        assert grid.shape == GRID_SHAPE
    assert (out_dir / "geometry.npz").exists()
