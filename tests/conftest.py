"""测试共用配置：固定 CPU、无界面绘图，并保证仓库根目录可导入。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 测试期间不弹窗、不使用 GPU
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from beta_cvae.utils import set_thread_env  # noqa: E402

set_thread_env()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """仓库根目录。"""
    return REPO_ROOT


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory) -> Path:
    """一次生成 smoke profile 合成数据，供多个测试复用。"""
    from beta_cvae.data.synthetic import generate_synthetic

    out = tmp_path_factory.mktemp("synthetic")
    generate_synthetic(out, profile="smoke", seed=0)
    return out
