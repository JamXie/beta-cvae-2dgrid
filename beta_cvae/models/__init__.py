"""模型包：基础 VAE、多分支、残差与最终 Beta-CVAE。"""

from .registry import DEFAULT_MODEL, MODEL_NAMES, build_model, get_spec, get_spec_for

__all__ = ["DEFAULT_MODEL", "MODEL_NAMES", "build_model", "get_spec", "get_spec_for"]
