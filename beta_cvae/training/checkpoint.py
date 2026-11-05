"""检查点保存与加载。

所有 `torch.load` 都显式传入 `map_location`，并优先使用
`weights_only=True`。为避免 PyTorch 1.x 的 weights-only 反序列化
不支持嵌套字典，配置与历史记录以 JSON 字符串保存。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    *,
    model_name: str,
    config: dict | None = None,
    history: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """保存模型参数与元信息，返回实际路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name": str(model_name),
        "model_state": dict(model.state_dict()),
        # 元信息统一转为 JSON 字符串，兼容 weights_only=True
        "config_json": json.dumps(config or {}, ensure_ascii=True),
        "history_json": json.dumps(history or {}, ensure_ascii=True),
        "extra_json": json.dumps(extra or {}, ensure_ascii=True),
    }

    torch.save(payload, path)
    return path


def load_checkpoint(path, device="cpu") -> dict:
    """加载检查点；优先 weights_only=True，并还原 JSON 元信息。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"检查点不存在：{path}")

    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # 旧版 PyTorch 没有 weights_only 参数
        payload = torch.load(path, map_location=device)
    except Exception as exc:  # noqa: BLE001 - 兼容旧检查点
        print(f"[warning] weights_only load failed ({exc}); fallback to full pickle")
        payload = torch.load(path, map_location=device, weights_only=False)

    for key in ("config", "history", "extra"):
        json_key = f"{key}_json"
        if json_key in payload:
            payload[key] = json.loads(payload.pop(json_key))
        else:
            payload.setdefault(key, {})

    return payload
