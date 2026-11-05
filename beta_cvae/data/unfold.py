"""3D 壳层图 -> 2D Grid 的展开与逆映射。

薄壁结构是闭合的壳层曲面。展开过程沿每一层（固定 z）的环向
连通路径切开展平，得到规则的 `[通道, 高, 宽]` 网格。
`node_positions` 记录每个 3D 节点落在 2D 网格中的 (x, z) 位置，
后续帧可以复用该映射，跳过昂贵的路径搜索。
"""

from __future__ import annotations

import numpy as np
import torch


def filter_edges(edge_list: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """提取当前层内部的边，并把全局节点索引映射为层内局部索引。"""
    layer_set = set(layer.tolist())
    edge_array = np.asarray(edge_list)
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("edge_list 应为形状 [E, 2] 的数组")

    edge_mask = np.array([i in layer_set and j in layer_set for i, j in edge_array])
    filtered_edges = edge_array[edge_mask]

    index_map = {node: idx for idx, node in enumerate(layer.tolist())}
    if filtered_edges.shape[0] == 0:
        # 返回 [2, 0]，与后续 `for i, j in layer_edges.T` 的用法保持一致
        return np.empty((2, 0), dtype=np.int64)

    local_edges = np.array([[index_map[i], index_map[j]] for i, j in filtered_edges]).T
    return local_edges


def extract_layer_path(layer_coords, layer_edges, start_point, end_point):
    """从起点与终点向两侧扩展到完整环向路径（辅助函数，保留原始语义）。"""
    adj: dict[int, list[int]] = {}
    for i, j in layer_edges.T:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)

    start_idx = int(np.argmin(np.linalg.norm(layer_coords - start_point, axis=1)))
    end_idx = int(np.argmin(np.linalg.norm(layer_coords - end_point, axis=1)))

    visited = set()

    forward_path = [start_idx]
    curr = start_idx
    prev = -1
    while True:
        visited.add(curr)
        neighbors = [n for n in adj.get(curr, []) if n != prev and n not in visited]
        if not neighbors or curr == end_idx:
            break
        prev, curr = curr, neighbors[0]
        forward_path.append(curr)

    backward_path = []
    curr = start_idx
    prev = -1
    while True:
        neighbors = [n for n in adj.get(curr, []) if n != prev and n not in visited]
        if not neighbors or curr == end_idx:
            break
        prev, curr = curr, neighbors[0]
        backward_path.append(curr)
        visited.add(curr)

    return backward_path[::-1] + forward_path


def find_intersection_candidate(layer_coords, plane_vector, z_value):
    """求解与展开平面方向对齐的起始/结束候选点（辅助函数）。"""
    center = np.mean(layer_coords[:, :2], axis=0)
    direction_2d = np.asarray(plane_vector[:2], dtype=np.float64)
    ray_directions = [direction_2d, -direction_2d]
    candidate_points = []

    for direction in ray_directions:
        min_dist = float("inf")
        closest_point = None
        for i in range(len(layer_coords)):
            p1 = layer_coords[i % len(layer_coords)][:2]
            p2 = layer_coords[(i + 1) % len(layer_coords)][:2]
            midpoint = (p1 + p2) / 2
            vec_to_mid = midpoint - center
            direction_unit = direction / (np.linalg.norm(direction) + 1e-6)
            if vec_to_mid.shape[0] != direction_unit.shape[0]:
                raise ValueError(
                    f"维度不匹配: vec_to_mid={vec_to_mid.shape}, direction_unit={direction_unit.shape}"
                )
            proj_len = np.dot(vec_to_mid, direction_unit)
            proj_point = center + proj_len * direction_unit
            dist = np.linalg.norm(proj_point - midpoint)
            if dist < min_dist:
                min_dist = dist
                closest_point = midpoint
        candidate_points.append(np.array([closest_point[0], closest_point[1], z_value]))

    return candidate_points[0], candidate_points[1]


def unfold_3d_grid_to_2d(current_nodes, edge_list, plane_vector=(0, -1, 0)):
    """将 3D 图展开为 2D Grid。

    参数：
    - `current_nodes`：`[N, 7]`，包含 xyz、位移 3 维与应力；
    - `edge_list`：`[E, 2]` 无向边；
    - `plane_vector`：展开平面的环向方向；源数据使用 `(0, -1, 0)`。

    返回：
    - `unfolded_grid`：`[4, W, H]`，通道为 dx/dy/dz/stress；
    - `node_positions`：`[N, 3]`，每行为 `[原始节点索引, x位置, z层]`。
    """
    coords = current_nodes[:, :3].cpu().numpy()
    features = current_nodes[:, 3:].cpu().numpy()
    edges = edge_list.cpu().numpy()

    z_values = np.unique(coords[:, 2])
    num_layers = len(z_values)
    layer_nodes = [np.where(coords[:, 2] == z)[0] for z in z_values]

    # 逐层确定环向起止点：从截面中心出发，沿 plane_vector 方向最近的边界节点
    start_points, end_points = [], []
    for z_idx, layer in enumerate(layer_nodes):
        layer_coords = coords[layer]
        z_val = z_values[z_idx]
        center = np.mean(layer_coords[:, :2], axis=0)

        direction_2d = np.array(plane_vector[:2], dtype=np.float64)
        direction_unit = direction_2d / (np.linalg.norm(direction_2d) + 1e-6)

        vecs = layer_coords[:, :2] - center
        vecs_norm = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs_unit = vecs / vecs_norm

        dot = np.sum(vecs_unit * direction_unit, axis=1)
        cross = direction_unit[0] * vecs_unit[:, 1] - direction_unit[1] * vecs_unit[:, 0]
        angles_deg = np.degrees(np.arctan2(cross, dot))
        angles_deg = (angles_deg + 360) % 360

        candidates_start = np.where(angles_deg > 0)[0]
        if len(candidates_start) == 0:
            candidates_start = np.arange(len(angles_deg))
        start_idx = candidates_start[np.argmin(angles_deg[candidates_start])]

        candidates_end = np.where(angles_deg > 180)[0]
        if len(candidates_end) == 0:
            candidates_end = np.arange(len(angles_deg))
        end_idx = candidates_end[np.argmin(np.abs(angles_deg[candidates_end] - 180))]

        start_points.append(np.array([layer_coords[start_idx][0], layer_coords[start_idx][1], z_val]))
        end_points.append(np.array([layer_coords[end_idx][0], layer_coords[end_idx][1], z_val]))

    W = max(len(nodes) for nodes in layer_nodes)
    H = num_layers
    F = features.shape[1]
    unfolded_grid = np.zeros((F, W, H), dtype=np.float32)

    node_positions = []
    for z_idx, layer in enumerate(layer_nodes):
        layer_coords = coords[layer]
        layer_features = features[layer]
        layer_edges = filter_edges(edges, layer)

        # 层内邻接表，邻居按索引升序，保证路径方向稳定
        adj: dict[int, list[int]] = {i: [] for i in range(len(layer))}
        for i, j in layer_edges.T:
            adj[i].append(j)
            adj[j].append(i)
        for k in adj:
            adj[k] = sorted(adj[k])

        start_idx = int(np.argmin(np.linalg.norm(layer_coords - start_points[z_idx], axis=1)))
        end_idx = int(np.argmin(np.linalg.norm(layer_coords - end_points[z_idx], axis=1)))

        # 正向：start -> end，含 start 不含 end
        forward_path = [start_idx]
        visited_forward = set(forward_path)
        curr = start_idx
        while True:
            neighbors = [n for n in adj[curr] if n not in visited_forward]
            if not neighbors or curr == end_idx:
                break
            curr = neighbors[0]
            forward_path.append(curr)
            visited_forward.add(curr)
        if forward_path[-1] == end_idx:
            forward_path.pop()

        # 反向：end -> start，含 end 不含 start
        backward_path = [end_idx]
        visited_backward = set(forward_path)
        visited_backward.add(end_idx)
        curr = end_idx
        while True:
            neighbors = [n for n in adj[curr] if n not in visited_backward]
            if not neighbors or curr == start_idx:
                break
            curr = neighbors[0]
            backward_path.append(curr)
            visited_backward.add(curr)

        # 合并路径：反向段 + 正向段，覆盖整圈且不重复
        full_path = backward_path + forward_path

        for x_pos, local_idx in enumerate(full_path):
            orig_node_idx = layer[local_idx]
            unfolded_grid[:, x_pos, z_idx] = layer_features[local_idx]
            node_positions.append([orig_node_idx, x_pos, z_idx])

    return torch.tensor(unfolded_grid), np.array(node_positions)


def fast_unfold_3d_grid_to_2d(current_nodes, node_positions):
    """使用预计算的 `node_positions` 将节点特征铺到 2D Grid。

    要求当前帧的节点集合与参考帧一致；不一致时应回退到完整展开。
    """
    features = current_nodes[:, 3:]
    F = features.shape[1]
    node_positions = np.asarray(node_positions)
    W = int(node_positions[:, 1].max()) + 1
    H = int(node_positions[:, 2].max()) + 1

    unfolded_grid = torch.zeros((F, W, H), dtype=features.dtype, device=features.device)
    for node_idx, x_pos, z_pos in node_positions:
        unfolded_grid[:, int(x_pos), int(z_pos)] = features[int(node_idx)]

    return unfolded_grid


def _2d_grid_to_3d(unfolded_grid, current_nodes, node_positions):
    """将 2D Grid 的通道值写回 3D 节点特征。"""
    unfolded_grid = torch.as_tensor(unfolded_grid)
    current_nodes = torch.as_tensor(current_nodes)
    node_positions = np.asarray(node_positions)

    F, W, H = unfolded_grid.shape
    recovered = torch.zeros_like(current_nodes)
    recovered[:, :3] = current_nodes[:, :3]

    for node_idx, x_pos, z_pos in node_positions:
        x_pos, z_pos = int(x_pos), int(z_pos)
        if 0 <= x_pos < W and 0 <= z_pos < H:
            recovered[node_idx, 3:] = unfolded_grid[:, x_pos, z_pos].to(recovered.dtype)

    return recovered
