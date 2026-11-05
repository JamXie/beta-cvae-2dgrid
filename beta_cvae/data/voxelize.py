"""有限元 CSV -> 体素 -> 3D 图节点。

CSV 列位置沿用原始工程约定（0 起始索引）：

- 位移：第 2 至第 4 列，即 `iloc[:, 2:5]`
- 节点坐标：第 9 至第 11 列，即 `iloc[:, 9:12]`
- Mises 应力：第 12 列，即 `iloc[:, 12:13]`

参考帧使用 `Degree0_Step0_*.csv` 的节点坐标确定体素占用，
保证不同时刻的体素拓扑一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def voxelize(
    node_data: np.ndarray,
    original_node_data: np.ndarray,
    displacement_data: np.ndarray,
    stress_data: np.ndarray,
    voxel_size: float,
    xyz_min: np.ndarray,
    xyz_max: np.ndarray,
) -> np.ndarray:
    """按参考帧节点坐标做体素化，并写入加权位移与应力。

    返回形状为 `[D, H, W, 5]` 的数组，通道含义：
    0 占位、1-3 位移、4 应力。`node_data` 参数保留以兼容原始签名，
    实际占用由 `original_node_data` 决定。
    """
    range_x = xyz_max[0] - xyz_min[0]
    range_y = xyz_max[1] - xyz_min[1]
    range_z = xyz_max[2] - xyz_min[2]

    range_voxel = np.ceil(np.array([range_x, range_y, range_z]) / voxel_size).astype(int)

    voxel_array = np.zeros((*range_voxel, 5), dtype=np.float64)
    # 每个体素内所有节点的权重累加，用于加权平均
    voxel_weight_sum = np.zeros((*range_voxel, 4), dtype=np.float64)

    for a in range(original_node_data.shape[0]):
        # 节点坐标 -> 体素索引，并裁剪到合法范围
        aa = int(np.floor((original_node_data[a, 0] - xyz_min[0]) / voxel_size))
        bb = int(np.floor((original_node_data[a, 1] - xyz_min[1]) / voxel_size))
        cc = int(np.floor((original_node_data[a, 2] - xyz_min[2]) / voxel_size))
        aa = min(max(0, aa), range_voxel[0] - 1)
        bb = min(max(0, bb), range_voxel[1] - 1)
        cc = min(max(0, cc), range_voxel[2] - 1)

        # 体素中心，用于计算距离权重
        voxel_center = np.array(
            [
                xyz_min[0] + (aa + 0.5) * voxel_size,
                xyz_min[1] + (bb + 0.5) * voxel_size,
                xyz_min[2] + (cc + 0.5) * voxel_size,
            ]
        )
        distance = np.linalg.norm(original_node_data[a] - voxel_center)
        weight = 1.0 / (distance + 1e-6)

        voxel_array[aa, bb, cc, 0] = 1
        voxel_array[aa, bb, cc, 1] += displacement_data[a, 0] * weight
        voxel_array[aa, bb, cc, 2] += displacement_data[a, 1] * weight
        voxel_array[aa, bb, cc, 3] += displacement_data[a, 2] * weight
        voxel_array[aa, bb, cc, 4] += stress_data[a, 0] * weight

        voxel_weight_sum[aa, bb, cc, 0] += weight
        voxel_weight_sum[aa, bb, cc, 1] += weight
        voxel_weight_sum[aa, bb, cc, 2] += weight
        voxel_weight_sum[aa, bb, cc, 3] += weight

    # 对非空体素按权重和归一化
    non_zero_voxels = voxel_weight_sum[:, :, :, 0] > 0
    voxel_array[non_zero_voxels, 1] /= voxel_weight_sum[non_zero_voxels, 0]
    voxel_array[non_zero_voxels, 2] /= voxel_weight_sum[non_zero_voxels, 1]
    voxel_array[non_zero_voxels, 3] /= voxel_weight_sum[non_zero_voxels, 2]
    voxel_array[non_zero_voxels, 4] /= voxel_weight_sum[non_zero_voxels, 3]

    return voxel_array


def center_and_transform_coordinates(node_data: np.ndarray) -> np.ndarray:
    """将节点坐标沿 xy 平面中心平移到新原点。

    与原始工程不同之处：这里先复制输入，避免参考坐标被重复平移。
    """
    node_data = np.array(node_data, dtype=np.float64, copy=True)
    x_min, x_max = np.min(node_data[:, 0]), np.max(node_data[:, 0])
    y_min, y_max = np.min(node_data[:, 1]), np.max(node_data[:, 1])

    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    node_data[:, 0] -= x_mid
    node_data[:, 1] -= y_mid
    return node_data


def read_csv_and_voxelize(
    original_path,
    file_path,
    voxel_size: float = 3.0,
    xyz_min=None,
    xyz_max=None,
) -> np.ndarray:
    """读取当前帧与参考帧 CSV，返回 `[D, H, W, 5]` 体素数组。"""
    data01 = pd.read_csv(original_path)
    data = pd.read_csv(file_path)

    # 原始导出在第二列有空行，用作结束标记
    data01 = data01.dropna(subset=[data01.columns[1]])
    data = data.dropna(subset=[data.columns[1]])

    node_data = data.iloc[:, 9:12].values
    displacement_data = data.iloc[:, 2:5].values
    stress_data = data.iloc[:, 12:13].values

    node_data = center_and_transform_coordinates(node_data)
    original_node_data = data01.iloc[:, 9:12].values
    original_node_data = center_and_transform_coordinates(original_node_data)

    if xyz_min is None:
        xyz_min = np.min(node_data, axis=0)
    if xyz_max is None:
        xyz_max = np.max(node_data, axis=0)

    return voxelize(
        node_data,
        original_node_data,
        displacement_data,
        stress_data,
        voxel_size,
        np.asarray(xyz_min, dtype=np.float64),
        np.asarray(xyz_max, dtype=np.float64),
    )


def voxel_array_to_grid(
    voxel_array: np.ndarray,
    disp_scale: float,
    stress_scale: float,
):
    """体素数组 -> 归一化 3D 图。

    先把 `[D, H, W, 5]` 转为 `[5, D, H, W]`，再按通道归一化，
    然后构建节点特征与六邻域边。返回 `(nodes[N,7], edges[E,2])`。
    """
    array = np.asarray(voxel_array, dtype=np.float32)
    if array.ndim != 4 or array.shape[-1] != 5:
        raise ValueError(f"体素数组应为 [D,H,W,5]，实际为 {array.shape}")

    # 通道前置，并只对位移/应力做归一化，占位通道保持 0/1
    array = np.transpose(array, (3, 0, 1, 2)).copy()
    array[1:4] /= disp_scale
    array[4] /= stress_scale

    return voxel_to_grid(array)


def voxel_to_grid(voxel_array, voxel_size: float = 1.0, xyz_min=None):
    """将 `[C, D, H, W]` 体素数据转换为节点特征与邻接边。

    返回：
    - `current_nodes`：`[N, 7]`，即 xyz + 位移 3 维 + 应力；
    - `edge_list`：`[E, 2]`，六邻域无向边的有序索引对。
    """
    if isinstance(voxel_array, torch.Tensor):
        array = voxel_array.detach().cpu().numpy()
    else:
        array = np.asarray(voxel_array)

    if array.ndim == 5:
        array = array[0]
    if array.ndim != 4:
        raise ValueError(f"体素数组应为 [C,D,H,W] 或 [B,C,D,H,W]，实际为 {array.shape}")

    _, D, H, W = array.shape
    if xyz_min is None:
        xyz_min = np.array([0.0, 0.0, 0.0])
    else:
        xyz_min = np.asarray(xyz_min, dtype=np.float64)

    node_idx_map: dict[tuple[int, int, int], int] = {}
    current_nodes: list[np.ndarray] = []

    # 通道 0 > 0.5 表示该体素被结构占据
    relevant_voxels = np.argwhere(array[0, ...] > 0.5)

    for idx, (i, j, k) in enumerate(relevant_voxels):
        x = xyz_min[0] + (i + 0.5) * voxel_size
        y = xyz_min[1] + (j + 0.5) * voxel_size
        z = xyz_min[2] + (k + 0.5) * voxel_size

        displacements = array[1:4, i, j, k]
        stress = array[4, i, j, k]

        node_feat = np.concatenate([[x, y, z], displacements, [stress]])
        current_nodes.append(node_feat)
        node_idx_map[(i, j, k)] = idx

    # 六邻域连边，使用有序元组避免重复
    edge_set: set[tuple[int, int]] = set()
    for (i, j, k) in relevant_voxels:
        current_idx = node_idx_map[(i, j, k)]
        potential_neighbors = [
            (i + 1, j, k),
            (i - 1, j, k),
            (i, j + 1, k),
            (i, j - 1, k),
            (i, j, k + 1),
            (i, j, k - 1),
        ]
        for neighbor in potential_neighbors:
            if neighbor in node_idx_map:
                neighbor_idx = node_idx_map[neighbor]
                edge_set.add(tuple(sorted((current_idx, neighbor_idx))))

    if not current_nodes:
        raise ValueError("体素数组中没有有效节点，请检查 xyz_min/xyz_max 与 voxel_size")

    edge_list = np.array(sorted(edge_set), dtype=np.int64)
    if edge_list.size == 0:
        edge_list = np.empty((0, 2), dtype=np.int64)

    nodes = torch.tensor(np.array(current_nodes), dtype=torch.float32)
    edges = torch.tensor(edge_list, dtype=torch.long)
    return nodes, edges.contiguous()
