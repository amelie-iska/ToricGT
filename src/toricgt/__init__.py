"""ToricGT research prototype package."""

from .random_order_lm import DenseRandomOrderToricLM, RandomOrderLMConfig
from .soft_moe import GraphTokenSoftMoE, PrefixCausalSoftMoE

__all__ = [
    "__version__",
    "DenseRandomOrderToricLM",
    "GraphTokenSoftMoE",
    "PrefixCausalSoftMoE",
    "RandomOrderLMConfig",
]

__version__ = "0.1.0"
