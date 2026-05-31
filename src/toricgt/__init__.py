"""ToricGT research prototype package."""

from .random_order_lm import DenseRandomOrderToricLM, RandomOrderLMConfig
from .soft_moe import GraphTokenSoftMoE, PrefixCausalSoftMoE
from .topological_reasoning import ReasoningTopologyConfig, reasoning_step_topology_loss
from .toric_geometry_tasks import LowRankToricGeometryProbe, ToricGeometryConfig

__all__ = [
    "__version__",
    "DenseRandomOrderToricLM",
    "GraphTokenSoftMoE",
    "LowRankToricGeometryProbe",
    "PrefixCausalSoftMoE",
    "RandomOrderLMConfig",
    "ReasoningTopologyConfig",
    "ToricGeometryConfig",
    "reasoning_step_topology_loss",
]

__version__ = "0.1.0"
