"""ToricGT research prototype package."""

from .random_order_lm import DenseRandomOrderToricLM, RandomOrderLMConfig
from .soft_moe import GraphTokenSoftMoE, PrefixCausalSoftMoE
from .combinatorial_toric_metrics import CombinatorialToricConfig, combinatorial_toric_cca_topology_loss
from .symbolic_multigraded_resolution import (
    SymbolicResolutionMetrics,
    cyclic_stanley_reisner_betti_rows,
    cyclic_stanley_reisner_generator_masks,
    cyclic_stanley_reisner_resolution_dict,
    cyclic_stanley_reisner_resolution_metrics,
    cyclic_taylor_multidegree_counts,
)
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
    "SymbolicResolutionMetrics",
    "TrajectoryMemoryConfig",
    "TrajectoryMemoryIndex",
    "TrajectoryMemoryRecord",
    "TrajectoryRetrievalHead",
    "ToricGeometryConfig",
    "combinatorial_toric_cca_topology_loss",
    "cyclic_stanley_reisner_betti_rows",
    "cyclic_stanley_reisner_generator_masks",
    "cyclic_stanley_reisner_resolution_dict",
    "cyclic_stanley_reisner_resolution_metrics",
    "cyclic_taylor_multidegree_counts",
    "reasoning_step_topology_loss",
]

__version__ = "0.1.0"
