"""2D Grid 数据集与训练/验证/测试划分。

文件名格式：`Degree{D}_Step{S}_F{F}.npy`，每帧形状 `[4, 80, 84]`。

划分规则与论文一致：

- 训练：角度为 5 的倍数，时间步为 5 的倍数；
- 验证：角度取自质数集合（不含 23），时间步为 5 的倍数；
- 测试/预测：默认角度 23，不参与训练与 RBFI 拟合。

原始工程中测试角度误取 5，与训练集重叠；这里改为 23，
并在构造数据集时断言三个集合互不交叉。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import (
    DEFAULT_TEST_ANGLE,
    GRID_SHAPE,
    TRAIN_DEGREE_MOD,
    TRAIN_STEP_MOD,
    VAL_PRIME_ANGLES,
)
from ..utils import parse_grid_filename


@dataclass(frozen=True)
class Sample:
    """一个 npy 帧的元数据。"""

    path: Path
    degree: int
    step: int
    force: float


def collect_samples(root: str | Path) -> list[Sample]:
    """递归收集 root 下所有符合命名规则的 npy 文件，按路径排序。"""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在：{root}")

    samples: list[Sample] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            parsed = parse_grid_filename(filename)
            if parsed is None:
                continue
            degree, step, force = parsed
            samples.append(Sample(Path(dirpath) / filename, degree, step, force))

    samples.sort(key=lambda s: (s.degree, s.step, s.path.name))
    return samples


def select_samples(
    samples: list[Sample],
    *,
    degrees: set[int] | None = None,
    degree_mod: int | None = None,
    steps: set[int] | None = None,
    step_mod: int | None = None,
) -> list[Sample]:
    """按角度集合/模数、时间步集合/模数筛选样本。"""
    selected = []
    for sample in samples:
        if degrees is not None and sample.degree not in degrees:
            continue
        if degree_mod is not None and sample.degree % degree_mod != 0:
            continue
        if steps is not None and sample.step not in steps:
            continue
        if step_mod is not None and sample.step % step_mod != 0:
            continue
        selected.append(sample)
    return selected


def check_split_disjoint(
    test_angle: int = DEFAULT_TEST_ANGLE,
    val_angles: tuple[int, ...] = VAL_PRIME_ANGLES,
    train_degree_mod: int = TRAIN_DEGREE_MOD,
) -> None:
    """确保测试角度既不在训练角度集合中，也不在验证角度集合中。"""
    if test_angle in val_angles:
        raise ValueError(f"测试角度 {test_angle} 出现在验证集中，会造成泄漏")
    if test_angle % train_degree_mod == 0:
        raise ValueError(f"测试角度 {test_angle} 是训练角度模数 {train_degree_mod} 的倍数，会造成泄漏")
    overlap = [angle for angle in val_angles if angle % train_degree_mod == 0]
    if overlap:
        raise ValueError(f"验证角度 {overlap} 与训练角度重叠")


def split_samples(
    root: str | Path,
    split: str,
    *,
    test_angle: int = DEFAULT_TEST_ANGLE,
    val_angles: tuple[int, ...] = VAL_PRIME_ANGLES,
    train_degree_mod: int = TRAIN_DEGREE_MOD,
    step_mod: int = TRAIN_STEP_MOD,
) -> list[Sample]:
    """返回指定划分的样本列表。"""
    check_split_disjoint(test_angle, tuple(val_angles), train_degree_mod)
    samples = collect_samples(root)

    if split == "train":
        result = select_samples(samples, degree_mod=train_degree_mod, step_mod=step_mod)
    elif split == "val":
        result = select_samples(samples, degrees=set(val_angles), step_mod=step_mod)
    elif split in ("test", "predict"):
        result = select_samples(samples, degrees={test_angle}, step_mod=step_mod)
    else:
        raise ValueError(f"未知划分 {split!r}，可选 train/val/test")

    if not result:
        raise ValueError(
            f"划分 {split!r} 为空：请检查数据目录 {root}、角度与时间步筛选条件"
        )
    return result


class NpyGridDataset(Dataset):
    """读取 2D Grid npy，返回 `(grid, degree, step)`。"""

    def __init__(self, samples: list[Sample], verify_shape: bool = True):
        if not samples:
            raise ValueError("样本列表为空")
        self.samples = list(samples)
        self.verify_shape = verify_shape

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        grid = np.load(sample.path, allow_pickle=False)
        grid = np.asarray(grid, dtype=np.float32)

        if self.verify_shape and grid.shape != GRID_SHAPE:
            raise ValueError(
                f"{sample.path} 形状为 {grid.shape}，期望 {GRID_SHAPE}，"
                "请先用 convert-csv 或 generate-synthetic 生成数据"
            )

        return grid, sample.degree, sample.step


def build_dataset(root: str | Path, split: str, **kwargs) -> NpyGridDataset:
    """构造指定划分的数据集。"""
    return NpyGridDataset(split_samples(root, split, **kwargs))
