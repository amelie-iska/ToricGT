"""ToricGT research prototype package."""

from .random_order_lm import DenseRandomOrderToricLM, RandomOrderLMConfig
from .soft_moe import GraphTokenSoftMoE, PrefixCausalSoftMoE
from .combinatorial_toric_metrics import CombinatorialToricConfig, combinatorial_toric_cca_topology_loss
from .topological_reasoning import ReasoningTopologyConfig, reasoning_step_topology_loss
from .toric_geometry_tasks import LowRankToricGeometryProbe, ToricGeometryConfig
from .trajectory_memory import TrajectoryMemoryConfig, TrajectoryMemoryIndex, TrajectoryMemoryRecord, TrajectoryRetrievalHead

__all__ = [
    "__version__",
    "CombinatorialToricConfig",
    "DenseRandomOrderToricLM",
    "GraphTokenSoftMoE",
    "LowRankToricGeometryProbe",
    "PrefixCausalSoftMoE",
    "RandomOrderLMConfig",
    "ReasoningTopologyConfig",
    "TrajectoryMemoryConfig",
    "TrajectoryMemoryIndex",
    "TrajectoryMemoryRecord",
    "TrajectoryRetrievalHead",
    "ToricGeometryConfig",
    "combinatorial_toric_cca_topology_loss",
    "reasoning_step_topology_loss",
]

__version__ = "0.1.0"
