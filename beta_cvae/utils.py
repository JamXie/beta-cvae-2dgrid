"""通用工具：随机种子、设备选择、目录创建与绘图后端。"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int = 0) -> None:
    """固定 Python、NumPy 与 PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # 仅数据生成时可以不装 torch
        pass


def resolve_device(name: str = "auto"):
    """解析设备名：auto 时优先使用 CUDA。"""
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA 设备，但当前环境不可用")
    return device


def ensure_dir(path: str | Path) -> Path:
    """创建目录并返回 Path。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def use_agg_backend() -> None:
    """强制使用无界面后端，避免服务器环境下阻塞。"""
    import matplotlib

    matplotlib.use("Agg")


def parse_grid_filename(name: str) -> tuple[int, int, float] | None:
    """从 npy 文件名解析 (Degree, Step, F)；不匹配时返回 None。"""
    from .constants import FILE_RE

    match = FILE_RE.fullmatch(Path(name).name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), float(match.group(3))


def set_thread_env() -> None:
    """限制部分 BLAS 线程数，避免小规模测试时的线程争用。

    只在用户未显式设置时生效，不覆盖用户配置。
    """
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")
