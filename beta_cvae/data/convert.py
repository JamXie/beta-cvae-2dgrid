"""真实有限元 CSV -> 2D Grid npy 的转换入口。

流程：
1. 参考帧（`Degree0_Step0_*.csv`）做完整展开，得到 `node_positions`；
2. 其余帧复用该映射做快速展开；
3. 若某帧节点数与参考帧不一致，打印警告并回退到完整展开。

输出与训练数据完全一致：`[4, 80, 84]` 的 npy 与 `geometry.npz`。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..constants import DISP_SCALE, GRID_SHAPE, STRESS_SCALE
from ..utils import ensure_dir
from .unfold import fast_unfold_3d_grid_to_2d, unfold_3d_grid_to_2d
from .voxelize import read_csv_and_voxelize, voxel_array_to_grid

CSV_RE = re.compile(r"Degree(\d+)_Step(\d+)_F([-0-9.]+)\.csv")


def _parse_frame(name: str) -> tuple[int, int, float] | None:
    match = CSV_RE.fullmatch(Path(name).name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), float(match.group(3))


def _to_2d_grid(voxel_array, node_positions=None, plane_vector=(0, -1, 0)):
    """体素 -> 3D 图 -> 2D Grid；可选复用参考帧映射。"""
    nodes, edges = voxel_array_to_grid(voxel_array, DISP_SCALE, STRESS_SCALE)

    if node_positions is not None and len(nodes) == len(node_positions):
        return fast_unfold_3d_grid_to_2d(nodes, node_positions), nodes, edges, node_positions, True

    grid, new_positions = unfold_3d_grid_to_2d(nodes, edges, plane_vector=plane_vector)
    return grid, nodes, edges, new_positions, False


def convert_csv_directory(
    input_dir,
    reference,
    output_dir,
    *,
    voxel_size: float = 3.0,
    xyz_min=(-150.0, -150.0, 0.0),
    xyz_max=(150.0, 150.0, 300.0),
    step_stride: int = 1,
    degree_stride: int = 1,
    limit: int | None = None,
    fast: bool = False,
    plane_vector=(0, -1, 0),
    save_geometry: bool = True,
) -> dict:
    """转换一个 work 目录下的所有 CSV 帧。返回统计信息。"""
    input_dir = Path(input_dir)
    reference = Path(reference)
    output_dir = ensure_dir(output_dir)

    if not reference.exists():
        raise FileNotFoundError(f"reference frame not found: {reference}")

    # 第一步：参考帧完整展开，建立节点到 2D 网格的映射
    ref_voxel = read_csv_and_voxelize(
        reference, reference, voxel_size, np.asarray(xyz_min), np.asarray(xyz_max)
    )
    ref_grid, ref_nodes, ref_edges, node_positions, _ = _to_2d_grid(
        ref_voxel, None, plane_vector=plane_vector
    )

    if ref_grid.shape != GRID_SHAPE:
        print(
            f"[warning] reference frame unfolded to {tuple(ref_grid.shape)}, "
            f"expected {GRID_SHAPE}; check voxel size and bounding box"
        )

    if save_geometry:
        # 建立 2D 网格位置 -> 3D 节点索引的映射，供三维可视化使用
        height, width = GRID_SHAPE[1], GRID_SHAPE[2]
        grid_index = np.full((height, width), -1, dtype=np.int64)
        for node_idx, x_pos, z_pos in node_positions:
            x_pos, z_pos = int(x_pos), int(z_pos)
            if 0 <= z_pos < height and 0 <= x_pos < width:
                grid_index[z_pos, x_pos] = int(node_idx)

        np.savez_compressed(
            output_dir / "geometry.npz",
            node_positions=np.asarray(node_positions, dtype=np.int64),
            edge_index=ref_edges.cpu().numpy(),
            reference_nodes=ref_nodes[:, :3].cpu().numpy(),
            grid_index=grid_index,
        )

    # 第二步：逐帧转换
    frames = []
    for path in sorted(input_dir.glob("*.csv")):
        info = _parse_frame(path.name)
        if info is None:
            continue
        degree, step, force = info
        if degree % degree_stride != 0 or step % step_stride != 0:
            continue
        frames.append((path, degree, step, force))

    if limit is not None:
        frames = frames[:limit]

    if not frames:
        raise ValueError(f"no matching CSV frames under {input_dir}")

    converted = []
    full_unfold_count = 0
    for path, degree, step, force in frames:
        voxel = read_csv_and_voxelize(
            reference, path, voxel_size, np.asarray(xyz_min), np.asarray(xyz_max)
        )

        if fast:
            grid, nodes, _, _, used_fast = _to_2d_grid(
                voxel, node_positions, plane_vector=plane_vector
            )
            if not used_fast:
                print(f"[warning] {path.name}: node count differs from reference, fell back to full unfold")
                full_unfold_count += 1
        else:
            grid, nodes, _, _, _ = _to_2d_grid(voxel, None, plane_vector=plane_vector)

        if grid.shape != GRID_SHAPE:
            print(f"[warning] {path.name}: shape {tuple(grid.shape)} differs from {GRID_SHAPE}")

        out_name = f"Degree{degree}_Step{step}_F{force}.npy"
        np.save(output_dir / out_name, grid.cpu().numpy().astype(np.float32))
        converted.append(out_name)

    meta = {
        "source": str(input_dir),
        "reference": str(reference),
        "output": str(output_dir),
        "voxel_size": voxel_size,
        "xyz_min": list(xyz_min),
        "xyz_max": list(xyz_max),
        "step_stride": step_stride,
        "degree_stride": degree_stride,
        "plane_vector": list(plane_vector),
        "fast": bool(fast),
        "num_frames": len(converted),
        "full_unfold_fallbacks": full_unfold_count,
        "grid_shape": list(GRID_SHAPE),
        "files": converted,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
