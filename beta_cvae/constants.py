"""全局常量：网格形状、填充/裁剪规则、归一化尺度与数据划分。

论文第 5 章中所有 300/200/500/45 之类的魔法数只在这里出现，
其他模块必须从这里导入，避免多处硬编码不一致。
"""

from __future__ import annotations

import re

# ========== 2D 网格 ==========
# 单帧 2D Grid 形状：[通道, 环向节点, 高度层]
# 通道 0-2 为 x/y/z 位移，通道 3 为 Mises 应力；
# 80 对应环向，84 对应高度方向，与网络 mask 的 80×84 有效区域一致
GRID_SHAPE: tuple[int, int, int] = (4, 80, 84)

# 网络需要 96×96 输入，因此在上下左右补零
# F.pad 的顺序为 (left, right, top, bottom)
PAD: tuple[int, int, int, int] = (6, 6, 8, 8)
PADDED_SHAPE: tuple[int, int] = (96, 96)

# 从 96×96 输出中裁回 80×84
# 高度 8:88（共 80），宽度 6:90（共 84）
CROP: tuple[slice, slice] = (slice(8, 88), slice(6, 90))

# ========== 归一化常数 ==========
# 位移除以 300（mm），应力除以 200（MPa），时间步除以 500，角度除以 45
DISP_SCALE: float = 300.0
STRESS_SCALE: float = 200.0
STEP_SCALE: float = 500.0
ANGLE_SCALE: float = 45.0

# ========== 网络默认结构 ==========
# 96×96 经过 4 次下采样得到 6×6，故 latent_Num=6
LATENT_NUM: int = 6
LATENT_DIM: int = 16

# ========== 数据划分 ==========
# 训练角度为 5 的倍数，训练时间步为 5 的倍数
TRAIN_DEGREE_MOD: int = 5
TRAIN_STEP_MOD: int = 5

# 验证角度取自质数集合。原工程中包含 23，但 23 同时被用作测试/预测角度，
# 会造成信息泄漏；这里将 23 从验证集移除。
VAL_PRIME_ANGLES: tuple[int, ...] = (7, 11, 13, 17, 19, 29, 31, 37, 41, 43)

# 测试与连续预测的默认角度（不参与训练与 RBFI 拟合）
DEFAULT_TEST_ANGLE: int = 23

# ========== 文件名解析 ==========
# 与原始数据命名一致，例如 Degree23_Step250_F-0.123.npy
FILE_PATTERN = r"Degree(\d+)_Step(\d+)_F([-0-9.]+)\.npy"
FILE_RE = re.compile(FILE_PATTERN)

# 连续预测的默认时间步
DEFAULT_PREDICT_STEPS: tuple[int, ...] = (0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500)
