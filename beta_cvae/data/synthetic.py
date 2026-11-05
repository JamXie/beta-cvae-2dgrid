"""合成数据生成器：用于克隆后的冒烟运行与单元测试。

生成三部分内容：

1. `npy/`：符合 `Degree{D}_Step{S}_F{F}.npy` 命名、形状 `[4, 80, 84]` 的网格，
   其中第 2 维为环向节点（80），第 3 维为高度层（84）；
2. `csv/work0/`：4 帧合成有限元 CSV，用于验证 `convert-csv`；
3. `geometry.npz`：参考节点坐标、边与 2D 网格映射，供三维可视化。

几何为一个 80 节点环向截面、84 层的矩形薄壁管，节点间距恰好 3 mm，
与体素尺寸一致，因此转换后仍得到 80×84 的展开网格。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..constants import (
    DEFAULT_TEST_ANGLE,
    DISP_SCALE,
    GRID_SHAPE,
    STEP_SCALE,
    STRESS_SCALE,
    TRAIN_DEGREE_MOD,
    TRAIN_STEP_MOD,
    VAL_PRIME_ANGLES,
)
from ..utils import ensure_dir, set_seed

# 展开网格为 [通道, 环向节点, 高度层]
NUM_RING = GRID_SHAPE[1]
NUM_LAYERS = GRID_SHAPE[2]

# 截面半宽/半高：周长 240、共 80 点，间距 3 mm；84 层，间距 3 mm
HALF_WIDTH = 40.0
HALF_HEIGHT = 20.0
LAYER_SPACING = 3.0
TOTAL_HEIGHT = (NUM_LAYERS - 1) * LAYER_SPACING


def ring_coordinates(n: int = NUM_RING) -> np.ndarray:
    """沿矩形截面周长等距取 n 个点，返回 `[n, 2]`。"""
    width = 2 * HALF_WIDTH
    height = 2 * HALF_HEIGHT
    perimeter = 2 * (width + height)
    s = np.arange(n) * (perimeter / n)

    coords = np.zeros((n, 2), dtype=np.float64)
    for i, value in enumerate(s):
        if value < width:  # 底边：从左到右
            coords[i] = [-HALF_WIDTH + value, -HALF_HEIGHT]
        elif value < width + height:  # 右边：从下到上
            coords[i] = [HALF_WIDTH, -HALF_HEIGHT + (value - width)]
        elif value < 2 * width + height:  # 顶边：从右到左
            coords[i] = [HALF_WIDTH - (value - width - height), HALF_HEIGHT]
        else:  # 左边：从上到下
            coords[i] = [-HALF_WIDTH, HALF_HEIGHT - (value - 2 * width - height)]
    return coords


def reference_geometry() -> dict:
    """构造参考帧的节点坐标、2D 网格映射与边。

    `grid_index` 与展开网格的 `[环向, 层]` 索引一致。
    """
    ring = ring_coordinates()
    layers = np.arange(NUM_LAYERS) * LAYER_SPACING

    num_nodes = NUM_LAYERS * NUM_RING
    nodes = np.zeros((num_nodes, 3), dtype=np.float64)
    node_positions = np.zeros((num_nodes, 3), dtype=np.int64)
    grid_index = np.full((NUM_RING, NUM_LAYERS), -1, dtype=np.int64)

    for h, z in enumerate(layers):
        for w in range(NUM_RING):
            node_id = h * NUM_RING + w
            nodes[node_id] = [ring[w, 0], ring[w, 1], z]
            # 与 unfold 的输出一致：[原始节点索引, x位置(环向), z层]
            node_positions[node_id] = [node_id, w, h]
            grid_index[w, h] = node_id

    # 边：同层环向相邻 + 相邻层同一环向位置
    edges = []
    for h in range(NUM_LAYERS):
        for w in range(NUM_RING):
            node_id = h * NUM_RING + w
            edges.append([node_id, h * NUM_RING + (w + 1) % NUM_RING])
            if h + 1 < NUM_LAYERS:
                edges.append([node_id, (h + 1) * NUM_RING + w])

    edges = np.asarray(edges, dtype=np.int64)
    edges = np.unique(np.sort(edges, axis=1), axis=0)

    return {
        "node_positions": node_positions,
        "edge_index": edges,
        "reference_nodes": nodes,
        "grid_index": grid_index,
    }


def synthetic_grid(degree: int, step: int) -> np.ndarray:
    """解析式生成单帧 `[4, 80, 84]` 网格，取值约在 [-1, 1]。"""
    # 环向 w 为第 1 个空间轴，高度层 h 为第 2 个空间轴
    w = np.arange(NUM_RING)[:, None] / NUM_RING
    h = np.linspace(0.0, 1.0, NUM_LAYERS)[None, :]
    angle = np.deg2rad(float(degree))
    t = float(step) / STEP_SCALE

    # 沿高度的包络，两端位移趋近于零
    envelope = np.sin(np.pi * h) ** 1.5

    dx = 0.75 * envelope * np.cos(2 * np.pi * (w - 0.15 * angle)) * (0.45 + 0.55 * np.cos(angle))
    dy = -0.65 * envelope * np.sin(2 * np.pi * (w - 0.15 * angle)) * (0.40 + 0.60 * t)
    dz = -0.85 * (h ** 1.3) * (0.50 + 0.50 * np.cos(angle)) * (0.50 + 0.50 * np.sin(2 * np.pi * w))
    stress = (
        np.exp(-((h - 0.55) / 0.22) ** 2)
        * (0.35 + 0.35 * np.sin(2 * np.pi * w + angle))
        * (0.30 + 0.70 * max(t, 0.15))
    )

    grid = np.stack(
        [
            np.broadcast_to(dx, (NUM_RING, NUM_LAYERS)),
            np.broadcast_to(dy, (NUM_RING, NUM_LAYERS)),
            np.broadcast_to(dz, (NUM_RING, NUM_LAYERS)),
            np.broadcast_to(stress, (NUM_RING, NUM_LAYERS)),
        ],
        axis=0,
    )
    return np.clip(grid, -1.0, 1.0).astype(np.float32)


def force_value(degree: int, step: int) -> float:
    """与帧对应的归一化压溃力（仅用于文件名，训练不使用）。"""
    return round(0.70 * np.sin(np.deg2rad(degree)) + 0.50 * step / STEP_SCALE - 0.20, 3)


def _write_csv(path: Path, grid: np.ndarray, reference: bool) -> None:
    """按原始列位置写 CSV：位移 2-4 列、坐标 9-11 列、应力第 12 列。"""
    ring = ring_coordinates()
    layers = np.arange(NUM_LAYERS) * LAYER_SPACING

    num_nodes = NUM_LAYERS * NUM_RING
    displacement = np.zeros((num_nodes, 3), dtype=np.float64)
    coordinates = np.zeros((num_nodes, 3), dtype=np.float64)
    stress = np.zeros((num_nodes, 1), dtype=np.float64)

    for h, z in enumerate(layers):
        for w in range(NUM_RING):
            node_id = h * NUM_RING + w
            coordinates[node_id] = [ring[w, 0], ring[w, 1], z]
            if not reference:
                displacement[node_id] = grid[:3, w, h] * DISP_SCALE
                stress[node_id, 0] = grid[3, w, h] * STRESS_SCALE

    data = np.zeros((num_nodes, 13), dtype=np.float64)
    data[:, 0] = np.arange(num_nodes)
    data[:, 1] = np.arange(1, num_nodes + 1)  # 原始导出中的节点编号，非空
    data[:, 2:5] = displacement
    data[:, 9:12] = coordinates
    data[:, 12:13] = stress

    header = [
        "Index", "NodeId", "U1", "U2", "U3", "R1", "R2", "R3",
        "Filler", "X", "Y", "Z", "Mises",
    ]
    np.savetxt(path, data, delimiter=",", header=",".join(header), comments="", fmt="%.6f")


def generate_synthetic(
    out_dir,
    *,
    profile: str = "smoke",
    seed: int = 0,
) -> dict:
    """生成合成 npy、CSV 与 geometry.npz。"""
    if profile not in ("smoke", "full"):
        raise ValueError("profile 只能是 'smoke' 或 'full'")

    set_seed(seed)
    out_dir = ensure_dir(out_dir)
    npy_dir = ensure_dir(out_dir / "npy")
    csv_dir = ensure_dir(out_dir / "csv" / "work0")

    steps = (0, 250, 500) if profile == "smoke" else tuple(range(0, 501, 50))
    train_degrees = list(range(0, 46, TRAIN_DEGREE_MOD))
    val_degrees = list(VAL_PRIME_ANGLES)
    test_degrees = [DEFAULT_TEST_ANGLE]

    counts = {"train": 0, "val": 0, "test": 0}
    for degree in train_degrees:
        for step in steps:
            force = force_value(degree, step)
            np.save(npy_dir / f"Degree{degree}_Step{step}_F{force}.npy", synthetic_grid(degree, step))
            counts["train"] += 1

    for degree in val_degrees:
        for step in steps:
            force = force_value(degree, step)
            np.save(npy_dir / f"Degree{degree}_Step{step}_F{force}.npy", synthetic_grid(degree, step))
            counts["val"] += 1

    for degree in test_degrees:
        for step in steps:
            force = force_value(degree, step)
            np.save(npy_dir / f"Degree{degree}_Step{step}_F{force}.npy", synthetic_grid(degree, step))
            counts["test"] += 1

    # 4 帧 CSV：参考帧 + 3 个变形帧
    reference_path = csv_dir / "Degree0_Step0_F-0.0.csv"
    _write_csv(reference_path, synthetic_grid(0, 0), reference=True)
    for degree, step in ((23, 0), (23, 250), (5, 500)):
        force = force_value(degree, step)
        _write_csv(
            csv_dir / f"Degree{degree}_Step{step}_F{force}.csv",
            synthetic_grid(degree, step),
            reference=False,
        )

    geometry = reference_geometry()
    np.savez_compressed(out_dir / "geometry.npz", **geometry)

    meta = {
        "profile": profile,
        "seed": seed,
        "grid_shape": list(GRID_SHAPE),
        "axis_order": ["ring", "layer"],
        "dtype": "float32",
        "channels": ["dx", "dy", "dz", "mises"],
        "scales": {"disp": DISP_SCALE, "stress": STRESS_SCALE, "step": STEP_SCALE},
        "train_degrees": train_degrees,
        "val_degrees": val_degrees,
        "test_angle": DEFAULT_TEST_ANGLE,
        "steps": list(steps),
        "train_degree_mod": TRAIN_DEGREE_MOD,
        "step_mod": TRAIN_STEP_MOD,
        "counts": counts,
        "csv_reference": str(reference_path.name),
        "note": "合成数据仅用于冒烟运行，不代表论文结果",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
