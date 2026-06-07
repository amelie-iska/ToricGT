"""Reasoning-trajectory memory and anticipative retrieval heads.

The memory layer is deliberately small.  It is not a retrieval-augmented
language model bolted onto ToricGT.  It stores compact graph-of-thought
trajectory summaries and trains a head to predict which prior trajectory would
be useful to consult.  Offline indices can later feed long-context ring
attention, while the in-batch head gives a cheap training signal today.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .derived_category_metrics import chain_complex_from_edges_np, derived_category_feature_summary
from .got_trajectory import got_dag_metrics, got_dag_summary_np


@dataclass(frozen=True)
class TrajectoryMemoryConfig:
    projection_dim: int = 128
    max_summary_points: int = 96
    teacher_temperature: float = 0.20
    retrieval_temperature: float = 0.20
    distill_weight: float = 0.25
    quality_weight: float = 0.10
    topology_weight: float = 0.20
    graphcg_weight: float = 0.30
    toric_weight: float = 0.20
    dag_weight: float = 0.20
    derived_weight: float = 0.20


@dataclass
class TrajectoryMemoryRecord:
    record_id: str
    key: list[float]
    value: dict[str, Any]
    dataset: str = ""
    task_family: str = ""
    quality: float = 0.0
    helper_k: float = 0.0
    topology: dict[str, float] | None = None
    toric: dict[str, float] | None = None
    trajectory_graph: dict[str, Any] | None = None
    derived_category: dict[str, Any] | None = None


def _safe_normalize_np(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(denom, 1e-8)


def summarize_trajectory_np(
    hidden: np.ndarray,
    *,
    positions: np.ndarray | None = None,
    losses: np.ndarray | None = None,
    edges: np.ndarray | None = None,
    theta: float = 0.6180339887498948,
    beta: float = 1.4142135623730951,
    max_points: int = 96,
) -> dict[str, Any]:
    """Return a compact CPU summary for a reasoning trajectory."""

    points = np.asarray(hidden, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0:
        return {
            "key": [],
            "quality": 0.0,
            "topology": {},
            "toric": {},
        }
    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, num=max_points).round().astype(int)
        points = points[idx]
        if positions is not None:
            positions = np.asarray(positions)[idx]
        if losses is not None:
            losses = np.asarray(losses)[idx]
        if edges is not None:
            edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
            old_to_new = {int(old): int(new) for new, old in enumerate(idx.tolist())}
            remapped: list[tuple[int, int]] = []
            for src, dst in edge_array:
                if int(src) in old_to_new and int(dst) in old_to_new:
                    remapped.append((old_to_new[int(src)], old_to_new[int(dst)]))
            edges = np.asarray(remapped, dtype=np.int64).reshape(-1, 2) if remapped else None
    centered = points - points.mean(axis=0, keepdims=True)
    unit = _safe_normalize_np(centered)
    pooled = unit.mean(axis=0)
    endpoint = unit[-1] - unit[0] if unit.shape[0] > 1 else np.zeros_like(pooled)
    velocity = np.diff(unit, axis=0) if unit.shape[0] > 1 else np.zeros((1, unit.shape[1]), dtype=np.float32)
    speed = np.linalg.norm(velocity, axis=-1)
    if unit.shape[0] > 2:
        accel = np.diff(velocity, axis=0)
        curvature = np.linalg.norm(accel, axis=-1).mean()
    else:
        curvature = 0.0
    if unit.shape[0] >= 3:
        dists = np.linalg.norm(unit[:, None, :] - unit[None, :, :], axis=-1)
        finite = dists[np.triu_indices(unit.shape[0], k=1)]
        radius = float(np.quantile(finite, 0.18)) if finite.size else 0.0
        adjacency = (dists <= radius).astype(np.float32)
        np.fill_diagonal(adjacency, 0.0)
        edge_density = float(adjacency.sum() / max(1, unit.shape[0] * (unit.shape[0] - 1)))
    else:
        radius = 0.0
        edge_density = 0.0
    pos = np.arange(unit.shape[0], dtype=np.float32) if positions is None else np.asarray(positions, dtype=np.float32)
    phase_u = np.stack([np.sin(2.0 * math.pi * theta * pos), np.cos(2.0 * math.pi * theta * pos)], axis=-1).mean(axis=0)
    phase_v = np.stack([np.sin(2.0 * math.pi * beta * pos), np.cos(2.0 * math.pi * beta * pos)], axis=-1).mean(axis=0)
    loss_arr = np.asarray(losses, dtype=np.float32) if losses is not None else np.zeros((unit.shape[0],), dtype=np.float32)
    quality = float(-np.mean(loss_arr)) if loss_arr.size else 0.0
    scalars = np.asarray(
        [
            float(speed.mean()) if speed.size else 0.0,
            float(speed.std()) if speed.size else 0.0,
            float(curvature),
            float(edge_density),
            float(radius),
            float(phase_u[0]),
            float(phase_u[1]),
            float(phase_v[0]),
            float(phase_v[1]),
            quality,
        ],
        dtype=np.float32,
    )
    dag = got_dag_summary_np(unit.shape[0], edges=edges)
    chain = chain_complex_from_edges_np(unit.shape[0], edges=edges, max_vertices=min(max_points, 8))
    dag_scalars = np.asarray(
        [
            float(dag.get("branch_count", 0.0)),
            float(dag.get("merge_count", 0.0)),
            float(dag.get("edge_count", 0.0)),
            float(dag.get("back_edge_fraction", 0.0)),
            float(dag.get("simplex_edge_density", 0.0)),
            float(dag.get("max_out_degree", 0.0)),
            float(dag.get("max_in_degree", 0.0)),
        ],
        dtype=np.float32,
    )
    key = np.concatenate([pooled, endpoint, scalars, dag_scalars], axis=0)
    return {
        "key": key.astype(float).tolist(),
        "quality": quality,
        "topology": {
            "speed_mean": float(scalars[0]),
            "speed_std": float(scalars[1]),
            "curvature": float(scalars[2]),
            "edge_density": float(scalars[3]),
            "radius": float(scalars[4]),
            "got_dag_branch_count": float(dag_scalars[0]),
            "got_dag_merge_count": float(dag_scalars[1]),
            "got_dag_edge_count": float(dag_scalars[2]),
            "got_dag_back_edge_fraction": float(dag_scalars[3]),
            "got_dag_simplex_edge_density": float(dag_scalars[4]),
            "got_dag_max_out_degree": float(dag_scalars[5]),
            "got_dag_max_in_degree": float(dag_scalars[6]),
        },
        "toric": {
            "phase_u_sin": float(phase_u[0]),
            "phase_u_cos": float(phase_u[1]),
            "phase_v_sin": float(phase_v[0]),
            "phase_v_cos": float(phase_v[1]),
        },
        "trajectory_graph": {
            "kind": "branch_merge_dag",
            "edges": np.asarray(edges, dtype=np.int64).reshape(-1, 2).astype(int).tolist()
            if edges is not None
            else [],
            **dag,
        },
        "derived_category": {
            "kind": "got_trajectory_chain_complex_summary",
            "chain_complex": chain,
        },
    }


class TrajectoryMemoryIndex:
    """A compact cosine-search index for stored reasoning trajectories."""

    def __init__(self) -> None:
        self.records: list[TrajectoryMemoryRecord] = []
        self._keys: np.ndarray | None = None

    def add(self, record: TrajectoryMemoryRecord) -> None:
        if not record.key:
            return
        self.records.append(record)
        self._keys = None

    def build(self) -> None:
        if not self.records:
            self._keys = np.zeros((0, 0), dtype=np.float32)
            return
        keys = np.asarray([record.key for record in self.records], dtype=np.float32)
        self._keys = _safe_normalize_np(keys)

    def search(
        self,
        query_key: list[float] | np.ndarray,
        *,
        top_k: int = 8,
        dataset: str | None = None,
        task_family: str | None = None,
    ) -> list[tuple[TrajectoryMemoryRecord, float]]:
        if self._keys is None:
            self.build()
        if self._keys is None or self._keys.shape[0] == 0:
            return []
        query = np.asarray(query_key, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self._keys.shape[1]:
            return []
        scores = self._keys @ _safe_normalize_np(query[None, :]).reshape(-1)
        order = np.argsort(-scores)
        out: list[tuple[TrajectoryMemoryRecord, float]] = []
        for idx in order.tolist():
            record = self.records[int(idx)]
            if dataset and record.dataset != dataset:
                continue
            if task_family and record.task_family != task_family:
                continue
            out.append((record, float(scores[int(idx)])))
            if len(out) >= top_k:
                break
        return out

    def save_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "TrajectoryMemoryIndex":
        index = cls()
        path = Path(path)
        if not path.exists():
            return index
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                index.add(TrajectoryMemoryRecord(**payload))
        index.build()
        return index


class TrajectoryRetrievalHead(nn.Module):
    """Train an anticipative retrieval score over trajectory summaries.

    The head uses in-batch candidates as a teacher-free approximation to the
    later offline memory task.  The teacher distribution favors trajectories
    with similar GraphCG chart coordinates, similar toric phase summaries,
    similar local topology, and better local NLL quality.
    """

    def __init__(self, d_model: int, config: TrajectoryMemoryConfig | None = None) -> None:
        super().__init__()
        self.config = config or TrajectoryMemoryConfig()
        projection_dim = int(self.config.projection_dim)
        self.query = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, projection_dim), nn.GELU(), nn.Linear(projection_dim, projection_dim))
        self.key = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, projection_dim), nn.GELU(), nn.Linear(projection_dim, projection_dim))
        self.quality_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def _summary_features(
        self,
        hidden: torch.Tensor,
        target_positions: torch.Tensor | None,
        graphcg_basis: torch.Tensor | None,
        trajectory_node_mask: torch.Tensor | None,
        trajectory_edge_index: torch.Tensor | None,
        trajectory_edge_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        h = hidden.float()
        if trajectory_node_mask is None:
            node_mask = torch.ones(h.shape[:2], device=h.device, dtype=torch.bool)
        else:
            node_mask = trajectory_node_mask.to(device=h.device, dtype=torch.bool)
        denom = node_mask.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
        pooled = (h * node_mask.unsqueeze(-1).to(h.dtype)).sum(dim=1) / denom
        last_index = node_mask.long().sum(dim=1).clamp_min(1) - 1
        first = h[:, 0, :]
        last = h[torch.arange(h.shape[0], device=h.device), last_index, :]
        endpoint = last - first if h.shape[1] > 1 else torch.zeros_like(pooled)
        summary = pooled + 0.25 * endpoint
        if h.shape[1] > 1:
            velocity = h[:, 1:, :] - h[:, :-1, :]
            speed = velocity.pow(2).mean(dim=-1).sqrt()
            speed_mean = speed.mean(dim=1)
            speed_std = speed.std(dim=1, unbiased=False)
        else:
            speed_mean = h.new_zeros((h.shape[0],))
            speed_std = h.new_zeros((h.shape[0],))
        if graphcg_basis is not None:
            basis = F.normalize(graphcg_basis.float(), dim=-1)
            chart = torch.matmul(F.normalize(pooled, dim=-1), basis.transpose(0, 1))
            chart_probs = torch.softmax(chart, dim=-1)
        else:
            chart_probs = F.normalize(pooled[:, : min(8, pooled.shape[-1])], dim=-1)
        if target_positions is None:
            pos = torch.arange(h.shape[1], device=h.device, dtype=torch.float32)[None, :].expand(h.shape[0], -1)
        else:
            pos = target_positions.float()
        theta = 0.6180339887498948
        beta = 1.4142135623730951
        toric = torch.stack(
            [
                torch.sin(2.0 * math.pi * theta * pos).mean(dim=1),
                torch.cos(2.0 * math.pi * theta * pos).mean(dim=1),
                torch.sin(2.0 * math.pi * beta * pos).mean(dim=1),
                torch.cos(2.0 * math.pi * beta * pos).mean(dim=1),
            ],
            dim=-1,
        )
        dag = got_dag_metrics(
            h,
            node_mask=node_mask,
            edge_index=trajectory_edge_index,
            edge_mask=trajectory_edge_mask,
        )
        topology = torch.stack(
            [
                speed_mean,
                speed_std,
                dag["got_dag_branch_count_batch"],
                dag["got_dag_merge_count_batch"],
                dag["got_dag_simplex_edge_density_batch"],
                dag["got_dag_triangle_density_batch"],
            ],
            dim=-1,
        )
        dag_feature = torch.stack(
            [
                dag["got_dag_branch_count_batch"],
                dag["got_dag_merge_count_batch"],
                dag["got_dag_back_edge_fraction_batch"],
                dag["got_dag_branch_diversity_batch"],
                dag["got_dag_merge_scatter_batch"],
                dag["got_dag_balance_residual_batch"],
            ],
            dim=-1,
        )
        derived_feature = derived_category_feature_summary(
            h,
            node_mask=node_mask,
            edge_index=trajectory_edge_index,
            edge_mask=trajectory_edge_mask,
        )
        return {
            "summary": summary,
            "chart": chart_probs,
            "toric": toric,
            "topology": topology,
            "dag": dag_feature,
            "derived": derived_feature,
        }

    def forward(
        self,
        hidden: torch.Tensor,
        target_positions: torch.Tensor | None,
        per_token_nll: torch.Tensor,
        *,
        graphcg_basis: torch.Tensor | None = None,
        trajectory_node_mask: torch.Tensor | None = None,
        trajectory_edge_index: torch.Tensor | None = None,
        trajectory_edge_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = hidden.shape[0]
        zero = hidden.new_zeros(())
        if batch < 2:
            return {
                "trajectory_memory_loss": zero,
                "trajectory_memory_ce": zero,
                "trajectory_memory_distill_loss": zero,
                "trajectory_memory_quality_loss": zero,
                "trajectory_memory_recall1": zero,
                "trajectory_memory_entropy": zero,
                "trajectory_memory_teacher_diag_prob": zero,
                "trajectory_memory_score_gap": zero,
                "trajectory_memory_dag_similarity": zero,
                "trajectory_memory_dag_branch_count": zero,
                "trajectory_memory_dag_merge_count": zero,
                "trajectory_memory_derived_similarity": zero,
                "trajectory_memory_derived_projective_dimension": zero,
                "trajectory_memory_derived_regularity": zero,
            }
        features = self._summary_features(
            hidden,
            target_positions,
            graphcg_basis,
            trajectory_node_mask,
            trajectory_edge_index,
            trajectory_edge_mask,
        )
        query = F.normalize(self.query(features["summary"].to(hidden.dtype)), dim=-1)
        key = F.normalize(self.key(features["summary"].to(hidden.dtype)), dim=-1)
        logits = torch.matmul(query, key.transpose(0, 1)) / max(float(self.config.retrieval_temperature), 1e-4)
        diag = torch.eye(batch, device=hidden.device, dtype=torch.bool)
        logits = logits.masked_fill(diag, -1e4)

        quality = -per_token_nll.detach().float().mean(dim=1)
        quality_z = (quality - quality.mean()) / (quality.std(unbiased=False) + 1e-6)
        with torch.no_grad():
            chart = F.normalize(features["chart"].float(), dim=-1)
            chart_sim = chart @ chart.transpose(0, 1)
            toric = F.normalize(features["toric"].float(), dim=-1)
            toric_sim = toric @ toric.transpose(0, 1)
            topo = features["topology"].float()
            topo_dist = torch.cdist(topo, topo, p=2)
            topo_sim = -topo_dist / (topo_dist.mean() + 1e-6)
            dag_feature = F.normalize(features["dag"].float(), dim=-1)
            dag_sim = dag_feature @ dag_feature.transpose(0, 1)
            derived_feature = F.normalize(features["derived"].float(), dim=-1)
            derived_sim = derived_feature @ derived_feature.transpose(0, 1)
            teacher = (
                float(self.config.graphcg_weight) * chart_sim
                + float(self.config.toric_weight) * toric_sim
                + float(self.config.topology_weight) * topo_sim
                + float(self.config.dag_weight) * dag_sim
                + float(self.config.derived_weight) * derived_sim
                + quality_z[None, :]
            )
            teacher = teacher.masked_fill(diag, -1e4) / max(float(self.config.teacher_temperature), 1e-4)
            labels = teacher.argmax(dim=-1)
            teacher_probs = torch.softmax(teacher, dim=-1)
        ce = F.cross_entropy(logits, labels)
        distill = F.kl_div(torch.log_softmax(logits, dim=-1), teacher_probs, reduction="batchmean")
        quality_pred = self.quality_head(features["summary"].to(hidden.dtype)).squeeze(-1).float()
        quality_loss = F.mse_loss(quality_pred, quality_z)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean() / math.log(max(2, batch - 1))
        pred = logits.argmax(dim=-1)
        top2 = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1).values
        gap = (top2[:, 0] - top2[:, -1]).mean() if top2.shape[-1] > 1 else zero
        loss = ce + float(self.config.distill_weight) * distill + float(self.config.quality_weight) * quality_loss
        return {
            "trajectory_memory_loss": loss,
            "trajectory_memory_ce": ce.detach(),
            "trajectory_memory_distill_loss": distill.detach(),
            "trajectory_memory_quality_loss": quality_loss.detach(),
            "trajectory_memory_recall1": (pred == labels).float().mean().detach(),
            "trajectory_memory_entropy": entropy.detach(),
            "trajectory_memory_teacher_diag_prob": teacher_probs.diagonal().mean().detach(),
            "trajectory_memory_score_gap": gap.detach(),
            "trajectory_memory_dag_similarity": dag_sim.masked_fill(diag, 0.0).mean().detach(),
            "trajectory_memory_dag_branch_count": features["dag"][:, 0].mean().detach(),
            "trajectory_memory_dag_merge_count": features["dag"][:, 1].mean().detach(),
            "trajectory_memory_derived_similarity": derived_sim.masked_fill(diag, 0.0).mean().detach(),
            "trajectory_memory_derived_projective_dimension": features["derived"][:, 7].mean().detach(),
            "trajectory_memory_derived_regularity": features["derived"][:, 8].mean().detach(),
        }
