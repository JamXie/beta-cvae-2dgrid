"""推理测试：RBFI 插值与误差指标。"""

from __future__ import annotations

import numpy as np

from beta_cvae.constants import DISP_SCALE
from beta_cvae.inference.rbfi import fit_rbfi, query_rbfi
from beta_cvae.metrics import grid_metrics


def test_rbfi_interpolates_training_points():
    """RBFI 在训练点上应近似精确插值。"""
    steps = np.array([0, 50, 100, 150, 200, 250], dtype=np.float64)
    degrees = np.array([0, 5, 10, 15, 20, 25], dtype=np.float64)
    points = np.stack([steps, degrees], axis=1)
    values = np.stack([steps / 500.0, degrees / 45.0, steps * degrees / 1e5], axis=1)

    rbf = fit_rbfi(points, values, epsilon=1.8)
    query = query_rbfi(rbf, steps, degrees)

    assert query.shape == values.shape
    assert np.allclose(query, values, atol=1e-6)


def test_rbfi_handles_scalar_query():
    """单点查询与广播查询都返回 `[N, latent]`。"""
    points = np.array([[0.0, 5.0], [100.0, 10.0], [200.0, 15.0], [300.0, 20.0]])
    values = points / np.array([500.0, 45.0])

    rbf = fit_rbfi(points, values)
    scalar = query_rbfi(rbf, 150.0, 12.0)
    broadcast = query_rbfi(rbf, np.array([100.0, 200.0]), np.array([10.0, 15.0]))

    assert scalar.shape == (1, 2)
    assert broadcast.shape == (2, 2)
    assert np.allclose(broadcast[0], values[1])


def test_metrics_zero_for_identical_grids():
    """预测与目标相同，所有误差为零。"""
    grid = np.random.RandomState(0).randn(4, 80, 84).astype(np.float32) * 0.1
    metrics = grid_metrics(grid, grid)
    for value in metrics.values():
        assert abs(value) < 1e-6


def test_metrics_constant_offset():
    """通道 0 整体偏移 0.01，对应位移 3 mm。"""
    target = np.zeros((4, 80, 84), dtype=np.float32)
    pred = target.copy()
    pred[0] += 0.01

    metrics = grid_metrics(pred, target)
    assert np.isclose(metrics["mae_x_mm"], 0.01 * DISP_SCALE)
    assert np.isclose(metrics["rmse_x_mm"], 0.01 * DISP_SCALE)
    assert np.isclose(metrics["euclidean_mae_mm"], 0.01 * DISP_SCALE)
    assert metrics["mae_y_mm"] == 0.0
    assert metrics["stress_mae_mpa"] == 0.0
