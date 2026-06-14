"""ToricGT research prototype package."""

from .random_order_lm import DenseRandomOrderToricLM, RandomOrderLMConfig
from .soft_moe import GraphTokenSoftMoE, PrefixCausalSoftMoE
from .combinatorial_toric_metrics import CombinatorialToricConfig, combinatorial_toric_cca_topology_loss
from .cas_certificates import (
    CASBackendInfo,
    CertificateCache,
    ToricTropicalCertificate,
    validate_certificate_payload,
)
from .cas_oracles import (
    Macaulay2TropicalOracle,
    SageToricOracle,
    cyclic_stanley_reisner_closed_form_certificate,
    discover_all_backends,
)
from .derived_category_metrics import (
    DerivedCategoryConfig,
    analogical_derived_category_loss,
    chain_complex_from_edges_np,
    derived_category_feature_summary,
    derived_category_objects_from_batch,
    projective_resolution_certificate,
)
from .got_trajectory import GoTDAGConfig, default_branch_merge_edges, got_dag_metrics, got_dag_summary_np
from .symbolic_multigraded_resolution import (
    SymbolicResolutionMetrics,
    cyclic_flag_face_rows,
    cyclic_stanley_reisner_betti_rows,
    cyclic_stanley_reisner_generator_masks,
    cyclic_stanley_reisner_resolution_certificate,
    cyclic_stanley_reisner_resolution_dict,
    cyclic_stanley_reisner_resolution_metrics,
    cyclic_taylor_dg_differential_rows,
    cyclic_taylor_dg_product_summary_rows,
    cyclic_taylor_fitting_entry_ideal_rows,
    cyclic_taylor_fitting_summary_rows,
    cyclic_taylor_multidegree_counts,
    cyclic_taylor_rank_rows,
)
from .topological_reasoning import ReasoningTopologyConfig, reasoning_step_topology_loss
from .toric_geometry_tasks import LowRankToricGeometryProbe, ToricGeometryConfig
from .trajectory_memory import TrajectoryMemoryConfig, TrajectoryMemoryIndex, TrajectoryMemoryRecord, TrajectoryRetrievalHead

__all__ = [
    "__version__",
    "CombinatorialToricConfig",
    "CASBackendInfo",
    "CertificateCache",
    "DerivedCategoryConfig",
    "DenseRandomOrderToricLM",
    "GraphTokenSoftMoE",
    "GoTDAGConfig",
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
    "ToricTropicalCertificate",
    "analogical_derived_category_loss",
    "chain_complex_from_edges_np",
    "combinatorial_toric_cca_topology_loss",
    "cyclic_flag_face_rows",
    "cyclic_stanley_reisner_betti_rows",
    "cyclic_stanley_reisner_closed_form_certificate",
    "cyclic_stanley_reisner_generator_masks",
    "cyclic_stanley_reisner_resolution_certificate",
    "cyclic_stanley_reisner_resolution_dict",
    "cyclic_stanley_reisner_resolution_metrics",
    "cyclic_taylor_dg_differential_rows",
    "cyclic_taylor_dg_product_summary_rows",
    "cyclic_taylor_fitting_entry_ideal_rows",
    "cyclic_taylor_fitting_summary_rows",
    "cyclic_taylor_multidegree_counts",
    "cyclic_taylor_rank_rows",
    "default_branch_merge_edges",
    "derived_category_feature_summary",
    "derived_category_objects_from_batch",
    "discover_all_backends",
    "got_dag_metrics",
    "got_dag_summary_np",
    "Macaulay2TropicalOracle",
    "projective_resolution_certificate",
    "reasoning_step_topology_loss",
    "SageToricOracle",
    "validate_certificate_payload",
]

__version__ = "0.1.0"
