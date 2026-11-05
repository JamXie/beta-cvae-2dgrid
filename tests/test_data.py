"""数据层测试：文件名解析、划分、形状、padding 与展开一致性。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from beta_cvae.constants import GRID_SHAPE
from beta_cvae.data.datasets import check_split_disjoint, collect_samples, split_samples
from beta_cvae.data.synthetic import ring_coordinates
from beta_cvae.data.unfold import (
    _2d_grid_to_3d,
    fast_unfold_3d_grid_to_2d,
    unfold_3d_grid_to_2d,
)
from beta_cvae.inference.decode import crop_prediction
from beta_cvae.training.trainer import pad_to_network
from beta_cvae.utils import parse_grid_filename


def test_parse_grid_filename():
    """符合规则的名称能被解析，其他名称返回 None。"""
    assert parse_grid_filename("Degree23_Step250_F-0.123.npy") == (23, 250, -0.123)
    assert parse_grid_filename("Degree0_Step0_F0.0.npy") == (0, 0, 0.0)
    assert parse_grid_filename("degree_23.npy") is None
    assert parse_grid_filename("Degree23_Step250.npy") is None


def _touch_frames(root, degrees, steps):
    """仅创建空文件，用于测试划分逻辑（不读取内容）。"""
    root.mkdir(parents=True, exist_ok=True)
    for degree in degrees:
        for step in steps:
            (root / f"Degree{degree}_Step{step}_F0.0.npy").touch()


def test_split_rules_and_disjointness(tmp_path):
    """训练/验证/测试按角度与时间步正确划分，且互不重叠。"""
    _touch_frames(tmp_path, degrees=(0, 5, 7, 23, 30), steps=(0, 5, 50))

    train = split_samples(tmp_path, "train")
    val = split_samples(tmp_path, "val")
    test = split_samples(tmp_path, "test")

    train_degrees = {sample.degree for sample in train}
    val_degrees = {sample.degree for sample in val}
    test_degrees = {sample.degree for sample in test}

    assert train_degrees == {0, 5, 30}
    assert val_degrees == {7}
    assert test_degrees == {23}
    assert train_degrees.isdisjoint(val_degrees | test_degrees)
    assert val_degrees.isdisjoint(test_degrees)
    assert all(sample.step % 5 == 0 for sample in train + val + test)


def test_split_rejects_leaked_test_angle():
    """测试角度 23 不能同时作为验证角度，也不能是训练模数的倍数。"""
    check_split_disjoint(test_angle=23, val_angles=(7, 11, 13))
    with pytest.raises(ValueError):
        check_split_disjoint(test_angle=23, val_angles=(7, 23))
    with pytest.raises(ValueError):
        check_split_disjoint(test_angle=25, val_angles=(7, 11))


def test_collect_samples_ignores_other_files(tmp_path):
    """非 npy 或命名不符的文件会被忽略。"""
    _touch_frames(tmp_path, degrees=(5,), steps=(0,))
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.npy").touch()
    samples = collect_samples(tmp_path)
    assert len(samples) == 1
    assert samples[0].degree == 5


def test_pad_crop_roundtrip():
    """`crop(pad(x)) == x`，填充不改变有效区域。"""
    grid = torch.randn(2, 4, *GRID_SHAPE[1:])
    padded = pad_to_network(grid)
    assert tuple(padded.shape) == (2, 4, 96, 96)
    assert torch.equal(crop_prediction(padded), grid)


def _small_shell_graph(num_ring=8, num_layers=4):
    """构造小型矩形管壳图：环向 8 点、4 层，节点特征可逆。"""
    ring = ring_coordinates(num_ring)
    layers = np.arange(num_layers, dtype=np.float64) * 3.0

    num_nodes = num_ring * num_layers
    nodes = np.zeros((num_nodes, 7), dtype=np.float32)
    for h, z in enumerate(layers):
        for w in range(num_ring):
            node_id = h * num_ring + w
            nodes[node_id, :3] = [ring[w, 0], ring[w, 1], z]
            nodes[node_id, 3:] = [w + 0.1 * h, h - 0.2 * w, w * h, 0.5 * w + h]

    edges = []
    for h in range(num_layers):
        for w in range(num_ring):
            node_id = h * num_ring + w
            edges.append([node_id, h * num_ring + (w + 1) % num_ring])
            if h + 1 < num_layers:
                edges.append([node_id, (h + 1) * num_ring + w])
    edges = np.unique(np.sort(np.asarray(edges, dtype=np.int64), axis=1), axis=0)
    return torch.tensor(nodes), torch.tensor(edges)


def test_fast_unfold_matches_full_unfold():
    """快速展开与完整展开在映射节点上完全一致。"""
    nodes, edges = _small_shell_graph()
    grid, node_positions = unfold_3d_grid_to_2d(nodes, edges)
    fast_grid = fast_unfold_3d_grid_to_2d(nodes, node_positions)

    assert fast_grid.shape == grid.shape
    assert torch.allclose(fast_grid, grid)


def test_grid_to_3d_recovers_features():
    """先展开再写回，节点特征与原始一致。"""
    nodes, edges = _small_shell_graph()
    grid, node_positions = unfold_3d_grid_to_2d(nodes, edges)
    recovered = _2d_grid_to_3d(grid, nodes, node_positions)

    assert torch.allclose(recovered[:, :3], nodes[:, :3])
    assert torch.allclose(recovered[:, 3:], nodes[:, 3:])
