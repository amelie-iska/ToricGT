"""ToricGT sidecar losses for the OAI Parameter-Golf baseline.

This module deliberately leaves the OAI SP1024 FineWeb language-model stream
untouched.  It consumes curated graph Parquet rows as an auxiliary stream and
trains graph/analogy/retrieval heads from the same GPT hidden states.
"""

from __future__ import annotations

import glob
import math
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .combinatorial_toric_metrics import CombinatorialToricConfig, combinatorial_toric_cca_topology_loss
from .koszul_persistence import KoszulPersistenceConfig, koszul_persistence_loss
from .trajectory_memory import TrajectoryMemoryConfig, TrajectoryRetrievalHead
from .toric_bgg import ToricBGGConfig, ToricBGGProbe
from .toric_geometry_tasks import LowRankToricGeometryProbe, ToricGeometryConfig
from .toric_vector_bundles import ToricVectorBundleConfig, ToricVectorBundleProbe


class GraphParquetTokenStream:
    """Stream curated graph rows as SP1024 token chunks for auxiliary training."""

    TEXT_COLUMNS = ("text", "question", "reasoning", "solution", "answer", "graph_json", "metadata_json")

    def __init__(self, pattern: str, sp: spm.SentencePieceProcessor, seq_len: int, batch_size: int):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No graph Parquet files found for pattern: {pattern}")
        self.sp = sp
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.file_idx = 0
        self.row_idx = 0
        self.rows: list[str] = []
        self.token_buffer: list[int] = []
        self._load_file()

    def _load_file(self) -> None:
        import pyarrow.parquet as pq

        path = self.files[self.file_idx]
        schema_names = set(pq.read_schema(path).names)
        columns = [name for name in self.TEXT_COLUMNS if name in schema_names]
        if not columns:
            raise ValueError(f"Graph Parquet shard has none of {self.TEXT_COLUMNS}: {path}")
        table = pq.read_table(path, columns=columns)
        rows: list[str] = []
        for idx in range(table.num_rows):
            pieces: list[str] = []
            for name in columns:
                value = table[name][idx].as_py()
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    pieces.append(text)
            if pieces:
                rows.append("\n".join(pieces))
        if not rows:
            raise ValueError(f"Graph Parquet shard produced no text rows: {path}")
        self.rows = rows
        self.row_idx = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self._load_file()

    def _append_next_row(self) -> None:
        if self.row_idx >= len(self.rows):
            self._advance_file()
        text = self.rows[self.row_idx]
        self.row_idx += 1
        ids = self.sp.encode(text, out_type=int)
        sep = self.sp.encode("\n\n", out_type=int)
        self.token_buffer.extend(int(v) for v in ids + sep)

    def next_batch(self, device: torch.device) -> tuple[Tensor, Tensor]:
        needed = self.batch_size * self.seq_len + 1
        while len(self.token_buffer) < needed:
            self._append_next_row()
        chunk = self.token_buffer[:needed]
        del self.token_buffer[: self.batch_size * self.seq_len]
        tokens = torch.tensor(chunk, dtype=torch.int64)
        x = tokens[:-1].reshape(self.batch_size, self.seq_len)
        y = tokens[1:].reshape(self.batch_size, self.seq_len)
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


class ToricGTSidecar(nn.Module):
    """Train graph, geometry, topology, category, and memory-retrieval heads."""

    def __init__(self, dim: int, args: Any):
        super().__init__()
        self.graphcg_basis = nn.Parameter(torch.empty(dim, dim))
        nn.init.orthogonal_(self.graphcg_basis)
        self.graph_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim, bias=False))
        self.edge_head = nn.Linear(dim, 1, bias=False)
        self.memory = TrajectoryRetrievalHead(
            dim,
            TrajectoryMemoryConfig(
                projection_dim=min(160, max(64, dim // 3)),
                teacher_temperature=0.20,
                retrieval_temperature=0.20,
                distill_weight=0.25,
                quality_weight=0.10,
                topology_weight=0.20,
                graphcg_weight=0.30,
                toric_weight=0.20,
                dag_weight=0.20,
                derived_weight=0.20,
                persistence_weight=0.20,
                persistence_max_points=48,
                persistence_landscape_layers=3,
                persistence_landscape_resolution=24,
                persistence_image_resolution=12,
            ),
        )
        self.graphcg_loss_weight = float(args.graphcg_loss_weight)
        self.analogy_loss_weight = float(args.analogy_loss_weight)
        self.tokengt_graph_loss_weight = float(args.tokengt_graph_loss_weight)
        self.trajectory_memory_loss_weight = float(args.trajectory_memory_loss_weight)
        self.toric_geometry_loss_weight = float(getattr(args, "toric_geometry_loss_weight", 0.0))
        self.toric_vector_bundle_loss_weight = float(getattr(args, "toric_vector_bundle_loss_weight", 0.0))
        self.toric_bgg_loss_weight = float(getattr(args, "toric_bgg_loss_weight", 0.0))
        self.koszul_persistence_loss_weight = float(getattr(args, "koszul_persistence_loss_weight", 0.0))
        self.combinatorial_toric_loss_weight = float(getattr(args, "combinatorial_toric_loss_weight", 0.0))
        self.compute_all_metrics = bool(int(getattr(args, "sidecar_compute_all_metrics", 1)))

        self.toric_geometry = LowRankToricGeometryProbe(
            dim,
            ToricGeometryConfig(
                enabled=True,
                num_exponents=int(getattr(args, "toric_geometry_num_exponents", 16)),
                exponent_dim=int(getattr(args, "toric_geometry_exponent_dim", 4)),
                probe_rank=int(getattr(args, "toric_geometry_probe_rank", 12)),
                quant_bits=int(getattr(args, "toric_geometry_quant_bits", 6)),
                max_positions=int(getattr(args, "toric_geometry_max_positions", 128)),
                fan_weight=float(getattr(args, "toric_geometry_fan_weight", 1.0)),
                bend_weight=float(getattr(args, "toric_geometry_bend_weight", 0.3)),
                binom_weight=float(getattr(args, "toric_geometry_binom_weight", 0.3)),
                moment_weight=float(getattr(args, "toric_geometry_moment_weight", 0.5)),
                coxeter_weight=float(getattr(args, "toric_geometry_coxeter_weight", 0.2)),
                braid_weight=float(getattr(args, "toric_geometry_braid_weight", 0.1)),
                leaf_weight=float(getattr(args, "toric_geometry_leaf_weight", 0.25)),
                cas_toric_ideal_certificate_path=str(getattr(args, "cas_toric_ideal_certificate_path", "")),
            ),
        )
        self.vector_bundle = ToricVectorBundleProbe(
            dim,
            ToricVectorBundleConfig(
                rank=int(getattr(args, "toric_vector_bundle_rank", 8)),
                num_one_dimensional_cones=int(
                    getattr(args, "toric_vector_bundle_num_one_dimensional_cones", 8)
                ),
                num_cones=int(getattr(args, "toric_vector_bundle_num_cones", 8)),
                filtration_levels=int(getattr(args, "toric_vector_bundle_filtration_levels", 3)),
                max_positions=int(getattr(args, "toric_vector_bundle_max_positions", 96)),
            ),
        )
        self.toric_bgg = ToricBGGProbe(
            dim,
            ToricBGGConfig(
                num_standard_tokens=int(getattr(args, "toric_bgg_num_standard_tokens", 8)),
                probe_rank=int(getattr(args, "toric_bgg_probe_rank", 8)),
                signature_dim=int(getattr(args, "toric_bgg_signature_dim", 16)),
                max_positions=int(getattr(args, "toric_bgg_max_positions", 64)),
            ),
        )
        self.koszul_cfg = KoszulPersistenceConfig(
            max_points=int(getattr(args, "koszul_max_points", 16)),
            max_windows=int(getattr(args, "koszul_max_windows", 2)),
            window_size=int(getattr(args, "koszul_window_size", 32)),
            step_stride=int(getattr(args, "koszul_step_stride", 16)),
            num_parameters=int(getattr(args, "koszul_num_parameters", 3)),
            temperature=float(getattr(args, "koszul_temperature", 0.12)),
            chart_exponents=int(getattr(args, "koszul_chart_exponents", 10)),
        )
        self.combinatorial_cfg = CombinatorialToricConfig(
            max_points=int(getattr(args, "combinatorial_toric_max_points", 16)),
            max_windows=int(getattr(args, "combinatorial_toric_max_windows", 2)),
            window_size=int(getattr(args, "combinatorial_toric_window_size", 32)),
            step_stride=int(getattr(args, "combinatorial_toric_step_stride", 16)),
            num_chambers=int(getattr(args, "combinatorial_toric_num_chambers", 8)),
            exponent_dim=int(getattr(args, "combinatorial_toric_exponent_dim", 4)),
        )

    @staticmethod
    def _byte_class_ids(tokens: Tensor) -> Tensor:
        cls = torch.full_like(tokens, 8)
        cls = torch.where((tokens >= 4) & (tokens < 128), torch.ones_like(cls), cls)
        cls = torch.where((tokens >= 128) & (tokens < 256), torch.full_like(cls, 2), cls)
        return torch.where(tokens >= 256, torch.full_like(cls, 3), cls)

    def _graphcg_losses(self, hidden: Tensor) -> dict[str, Tensor]:
        h = F.normalize(hidden.float().reshape(-1, hidden.shape[-1]), dim=-1)
        if h.shape[0] > 256:
            idx = torch.linspace(0, h.shape[0] - 1, steps=256, device=h.device).long()
            h = h.index_select(0, idx)
        basis = F.normalize(self.graphcg_basis.float(), dim=-1)
        coords = h @ basis.T
        eye = torch.eye(basis.shape[0], device=h.device, dtype=basis.dtype)
        gram = basis @ basis.T
        offdiag = max(1, basis.shape[0] * (basis.shape[0] - 1))
        orthogonal = (gram - eye).pow(2).sum() / offdiag
        centered = coords - coords.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / max(1, centered.shape[0] - 1)
        diag = cov.diagonal().clamp_min(1e-6)
        corr = cov / torch.sqrt(diag[:, None] * diag[None, :])
        covariance = (corr - eye).pow(2).sum() / offdiag
        axis_probs = torch.softmax(coords.abs() / 0.20, dim=-1)
        entropy = -(axis_probs * axis_probs.clamp_min(1e-8).log()).sum(dim=-1).mean() / math.log(max(2, basis.shape[0]))
        loss = orthogonal + 0.05 * covariance + 0.01 * entropy
        return {
            "graphcg_loss": loss,
            "graphcg_orthogonal_loss": orthogonal.detach(),
            "graphcg_covariance_loss": covariance.detach(),
            "graphcg_axis_entropy": entropy.detach(),
            "graphcg_chart_dim": hidden.new_tensor(float(basis.shape[0])).detach(),
        }

    def _analogy_losses(self, hidden: Tensor, targets: Tensor) -> dict[str, Tensor]:
        zero = hidden.float().sum() * 0.0
        if hidden.shape[1] < 3:
            return {"analogy_lattice_loss": zero, "analogy_relation_groups": zero.detach()}
        relation = F.normalize((hidden[:, 1:, :].float() - hidden[:, :-1, :].float()).reshape(-1, hidden.shape[-1]), dim=-1)
        classes = self._byte_class_ids(targets)
        keys = (classes[:, :-1] * 8 + classes[:, 1:]).reshape(-1)
        if relation.shape[0] > 512:
            idx = torch.linspace(0, relation.shape[0] - 1, steps=512, device=relation.device).long()
            relation = relation.index_select(0, idx)
            keys = keys.index_select(0, idx)
        unique, inverse, counts = torch.unique(keys, return_inverse=True, return_counts=True)
        repeated = counts[inverse] > 1
        if bool(repeated.any()):
            sums = relation.new_zeros(unique.shape[0], relation.shape[-1])
            sums.index_add_(0, inverse[repeated], relation[repeated])
            means = F.normalize(sums / counts.clamp_min(1).to(relation.dtype).unsqueeze(-1), dim=-1)
            functor = (relation[repeated] - means[inverse[repeated]]).pow(2).mean()
            groups = (counts > 1).to(relation.dtype).sum()
        else:
            functor = zero
            groups = zero
        basis = F.normalize(self.graphcg_basis.float(), dim=-1)
        coords = relation @ basis.T
        probs = torch.softmax(coords.abs() / 0.20, dim=-1)
        reconstructed = F.normalize((probs * coords.sign()) @ basis, dim=-1)
        basis_loss = (1.0 - (relation * reconstructed).sum(dim=-1)).mean()
        margin = (coords.abs().topk(2, dim=-1).values.diff(dim=-1).abs().mean() if coords.shape[-1] > 1 else coords.abs().mean())
        return {
            "analogy_lattice_loss": functor + 0.5 * basis_loss,
            "analogy_functor_loss": functor.detach(),
            "analogy_basis_loss": basis_loss.detach(),
            "analogy_lattice_margin": margin.detach(),
            "analogy_relation_groups": groups.detach(),
        }

    def _tokengt_graph_losses(self, hidden: Tensor, targets: Tensor) -> dict[str, Tensor]:
        h = self.graph_proj(hidden).float()
        bsz, seqlen, _ = h.shape
        n = min(seqlen, 128)
        if n < 4:
            zero = h.sum() * 0.0
            return {"tokengt_graph_loss": zero, "tokengt_graph_edge_density": zero.detach()}
        idx = torch.linspace(0, seqlen - 1, steps=n, device=h.device).round().long()
        nodes = F.normalize(h.index_select(1, idx), dim=-1)
        tgt = targets.index_select(1, idx)
        pos = idx.float()
        dist = (pos[:, None] - pos[None, :]).abs()
        eye = torch.eye(n, device=h.device, dtype=torch.bool)
        causal = (dist <= 2.0) & (pos[None, :] <= pos[:, None]) & ~eye
        same_class = self._byte_class_ids(tgt)[:, :, None] == self._byte_class_ids(tgt)[:, None, :]
        logits = torch.matmul(nodes, nodes.transpose(1, 2)) / 0.25
        mask = (~eye).to(logits.dtype)[None, :, :]
        edge_target = causal.unsqueeze(0).expand(bsz, -1, -1).to(logits.dtype)
        byte_target = same_class.to(logits.dtype) * mask
        edge_bce = F.binary_cross_entropy_with_logits(logits, edge_target, weight=mask, reduction="sum") / mask.sum().clamp_min(1.0) / bsz
        byte_bce = F.binary_cross_entropy_with_logits(logits, byte_target, weight=mask, reduction="sum") / mask.sum().clamp_min(1.0) / bsz
        projected_edges = torch.sigmoid(self.edge_head(nodes[:, 1:, :] - nodes[:, :-1, :])).mean()
        loss = edge_bce + 0.25 * byte_bce + 0.05 * (1.0 - projected_edges).pow(2)
        return {
            "tokengt_graph_loss": loss,
            "tokengt_graph_edge_bce": edge_bce.detach(),
            "tokengt_graph_byte_class_loss": byte_bce.detach(),
            "tokengt_graph_edge_density": edge_target.mean().detach(),
            "tokengt_graph_causal_edge_fraction": edge_target.mean().detach(),
        }

    def forward(self, hidden: Tensor, targets: Tensor, positions: Tensor, per_token_nll: Tensor) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        out.update(self._graphcg_losses(hidden))
        out.update(self._analogy_losses(hidden, targets))
        out.update(self._tokengt_graph_losses(hidden, targets))
        out.update(self.memory(hidden, positions, per_token_nll, graphcg_basis=self.graphcg_basis))
        run_all = bool(self.compute_all_metrics)
        if run_all or self.toric_geometry_loss_weight != 0.0:
            out.update(self.toric_geometry(hidden, positions, targets))
        if run_all or self.toric_vector_bundle_loss_weight != 0.0:
            out.update(self.vector_bundle(hidden, positions, targets))
        if run_all or self.toric_bgg_loss_weight != 0.0:
            out.update(self.toric_bgg(hidden, positions, targets))
        if run_all or self.koszul_persistence_loss_weight != 0.0:
            out.update(koszul_persistence_loss(hidden, positions, config=self.koszul_cfg))
        if run_all or self.combinatorial_toric_loss_weight != 0.0:
            out.update(combinatorial_toric_cca_topology_loss(hidden, positions, config=self.combinatorial_cfg))
        total = hidden.new_zeros(())
        total = total + self.graphcg_loss_weight * out["graphcg_loss"]
        total = total + self.analogy_loss_weight * out["analogy_lattice_loss"]
        total = total + self.tokengt_graph_loss_weight * out["tokengt_graph_loss"]
        total = total + self.trajectory_memory_loss_weight * out["trajectory_memory_loss"]
        total = total + self.toric_geometry_loss_weight * out.get("toric_geometry_loss", total.new_zeros(()))
        total = total + self.toric_vector_bundle_loss_weight * out.get(
            "toric_vector_bundle_1d_cone_ce_loss", total.new_zeros(())
        )
        total = total + self.toric_bgg_loss_weight * out.get("toric_bgg_loss", total.new_zeros(()))
        total = total + self.koszul_persistence_loss_weight * out.get(
            "koszul_persistence_loss", total.new_zeros(())
        )
        total = total + self.combinatorial_toric_loss_weight * out.get(
            "toric_cca_topology_loss", total.new_zeros(())
        )
        out["toric_geometry_active"] = total.new_tensor(float(self.toric_geometry_loss_weight != 0.0))
        out["toric_vector_bundle_1d_cone_ce_active"] = total.new_tensor(
            float(self.toric_vector_bundle_loss_weight != 0.0)
        )
        out["toric_bgg_active"] = total.new_tensor(float(self.toric_bgg_loss_weight != 0.0))
        out["koszul_persistence_active"] = total.new_tensor(float(self.koszul_persistence_loss_weight != 0.0))
        out["combinatorial_toric_active"] = total.new_tensor(float(self.combinatorial_toric_loss_weight != 0.0))
        out["sidecar_compute_all_metrics"] = total.new_tensor(float(self.compute_all_metrics))
        out["toricgt_sidecar_loss"] = total
        return out
