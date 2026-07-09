"""Embedding-space Forest-of-Thought training and trace head.

This module adapts Forest-of-Thought (FoT) from textual test-time prompting to
ToricGT's BPB-facing hidden states.  The reference FoT algorithm runs several
reasoning trees, sparsely activates useful trees, performs self-correction,
uses UCB-like exploration, and commits through forest consensus.  Here those
operations are represented as differentiable objectives over selected hidden
states, plus an inference/export trace with explicit tree, branch, correction,
and consensus edges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class EmbeddingFoTConfig:
    dim: int
    num_trees: int = 4
    max_depth: int = 5
    branching: int = 4
    topk_trees: int = 2
    hidden_dim: int = 192
    max_positions: int = 192
    consensus_buckets: int = 64
    correction_scale: float = 0.08
    ucb_exploration: float = 1.25
    temperature: float = 0.7
    sparse_weight: float = 1.0
    ucb_weight: float = 0.5
    correction_weight: float = 0.5
    consensus_weight: float = 0.75
    tb_weight: float = 1.0
    subtb_weight: float = 0.0
    complexity_weight: float = 0.05
    reward_advanced_bonus: float = 0.0
    reward_mode: str = "bpb_delta"
    bpb_delta_weight: float = 1.0
    reward_graph_weight: float = 0.10
    reward_consensus_weight: float = 0.20
    reward_complexity_weight: float = 0.02
    reward_floor: float = 1.0e-4


class EmbeddingFoTOutput(dict):
    """Dictionary output with a typed name for readability."""


class EmbeddingForestOfThoughtHead(nn.Module):
    """Differentiable FoT head over OAI baseline hidden states.

    The head views selected hidden positions as nodes in several interleaved
    reasoning trees.  It trains sparse activation, UCB-like expansion,
    self-correction, consensus, and trajectory-balance objectives using the
    local negative log likelihood as a dense BPB-native reward.
    """

    def __init__(self, config: EmbeddingFoTConfig) -> None:
        super().__init__()
        if config.dim <= 0:
            raise ValueError("EmbeddingFoTConfig.dim must be positive")
        if config.num_trees < 1:
            raise ValueError("OAI_FOT_NUM_TREES must be at least 1")
        if config.branching < 2:
            raise ValueError("OAI_FOT_BRANCHING must be at least 2")
        if config.max_depth < 1:
            raise ValueError("OAI_FOT_MAX_DEPTH must be at least 1")
        if config.consensus_buckets < 2:
            raise ValueError("OAI_FOT_CONSENSUS_BUCKETS must be at least 2")
        self.config = config
        d = int(config.dim)
        h = int(config.hidden_dim)
        self.node_norm = nn.LayerNorm(d)
        self.activation_head = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, 1))
        self.value_head = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, 1))
        self.forward_policy = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, config.branching))
        self.backward_policy = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, config.branching))
        self.correction_head = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))
        self.consensus_head = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, config.consensus_buckets))
        self.flow_head = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, 1))
        self.log_z = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def _selected_indices(self, length: int, device: torch.device) -> Tensor:
        cfg = self.config
        tree_budget = int(cfg.num_trees) * int(cfg.max_depth) * max(1, int(cfg.branching))
        max_positions = max(2, min(int(cfg.max_positions), tree_budget, int(length)))
        if max_positions >= length:
            return torch.arange(length, device=device, dtype=torch.long)
        return torch.linspace(0, length - 1, max_positions, device=device).round().long().unique(sorted=True)

    def _tree_reduce_mean(self, values: Tensor, tree_ids: Tensor, num_trees: int) -> Tensor:
        # values: [B, S] or [B, S, C]
        if values.ndim == 2:
            out = values.new_zeros(values.shape[0], num_trees)
            counts = values.new_zeros(num_trees).scatter_add_(0, tree_ids, torch.ones_like(tree_ids, dtype=values.dtype))
            out.scatter_add_(1, tree_ids.unsqueeze(0).expand(values.shape[0], -1), values)
            return out / counts.clamp_min(1.0).unsqueeze(0)
        out = values.new_zeros(values.shape[0], num_trees, values.shape[-1])
        counts = values.new_zeros(num_trees).scatter_add_(0, tree_ids, torch.ones_like(tree_ids, dtype=values.dtype))
        expand_ids = tree_ids.view(1, -1, 1).expand(values.shape[0], -1, values.shape[-1])
        out.scatter_add_(1, expand_ids, values)
        return out / counts.clamp_min(1.0).view(1, -1, 1)

    def _tree_last_indices(self, tree_ids: Tensor, num_trees: int) -> Tensor:
        last: list[int] = []
        for tree_id in range(num_trees):
            where = torch.nonzero(tree_ids == tree_id, as_tuple=False).flatten()
            last.append(int(where[-1].item()) if where.numel() else 0)
        return torch.tensor(last, device=tree_ids.device, dtype=torch.long)

    def _forest_topology(self, n_nodes: int, device: torch.device) -> dict[str, Tensor]:
        """Build a sparse FoT topology over selected hidden positions.

        The selected positions form ``num_trees`` interleaved trees.  Within a
        tree, nodes are arranged by depth and branch slot.  This is a bounded
        beam-style forest rather than a full exponential tree, which keeps the
        training/inference trace compatible with fixed context budgets while
        still exposing the FoT operations: branch expansion, self-correction,
        UCB transition scoring, and consensus over tree leaves.
        """

        num_trees = max(1, min(int(self.config.num_trees), int(n_nodes)))
        local = torch.arange(n_nodes, device=device, dtype=torch.long)
        tree_ids = torch.remainder(local, num_trees)
        tree_slots = torch.div(local, num_trees, rounding_mode="floor")
        branch_count = max(1, int(self.config.branching))
        depth_ids = torch.div(tree_slots, branch_count, rounding_mode="floor").clamp(max=max(0, int(self.config.max_depth) - 1))
        branch_ids = torch.remainder(tree_slots, branch_count)

        parent_by_key: dict[tuple[int, int, int], int] = {}
        latest_by_tree: dict[int, int] = {}
        parent_edges: list[tuple[int, int]] = []
        edge_actions: list[int] = []
        for node_id in range(n_nodes):
            tree = int(tree_ids[node_id].item())
            depth = int(depth_ids[node_id].item())
            branch = int(branch_ids[node_id].item())
            parent: int | None = None
            if depth > 0:
                parent = parent_by_key.get((tree, depth - 1, branch))
                if parent is None:
                    parent = latest_by_tree.get(tree)
            if parent is not None:
                parent_edges.append((parent, node_id))
                edge_actions.append(branch)
            parent_by_key[(tree, depth, branch)] = node_id
            latest_by_tree[tree] = node_id

        if parent_edges:
            edge_index = torch.tensor(parent_edges, device=device, dtype=torch.long)
            edge_actions_tensor = torch.tensor(edge_actions, device=device, dtype=torch.long)
        else:
            edge_index = torch.empty(0, 2, device=device, dtype=torch.long)
            edge_actions_tensor = torch.empty(0, device=device, dtype=torch.long)

        parent_nodes = edge_index[:, 0] if edge_index.numel() else torch.empty(0, device=device, dtype=torch.long)
        child_nodes = edge_index[:, 1] if edge_index.numel() else torch.empty(0, device=device, dtype=torch.long)
        has_child = torch.zeros(n_nodes, device=device, dtype=torch.bool)
        if parent_nodes.numel():
            has_child[parent_nodes] = True
        leaf_indices: list[int] = []
        for tree in range(num_trees):
            where = torch.nonzero((tree_ids == tree) & (~has_child), as_tuple=False).flatten()
            if where.numel() == 0:
                where = torch.nonzero(tree_ids == tree, as_tuple=False).flatten()
            leaf_indices.append(int(where[-1].item()) if where.numel() else 0)

        return {
            "num_trees": torch.tensor(num_trees, device=device, dtype=torch.long),
            "tree_ids": tree_ids,
            "depth_ids": depth_ids,
            "branch_ids": branch_ids,
            "edge_index": edge_index,
            "edge_actions": edge_actions_tensor,
            "parent_nodes": parent_nodes,
            "child_nodes": child_nodes,
            "leaf_indices": torch.tensor(leaf_indices, device=device, dtype=torch.long),
        }

    def forward(
        self,
        hidden: Tensor,
        target_ids: Tensor,
        per_token_nll: Tensor,
        *,
        advanced_signal: Tensor | None = None,
        target_byte_lengths: Tensor | None = None,
        lm_head_weight: Tensor | None = None,
        logit_softcap: float = 30.0,
        temperature_multiplier: float = 1.0,
        ucb_multiplier: float = 1.0,
        sparse_multiplier: float = 1.0,
    ) -> EmbeddingFoTOutput:
        cfg = self.config
        zero = hidden.new_zeros(())
        if hidden.ndim != 3 or hidden.shape[1] < 2:
            return EmbeddingFoTOutput({"oai_fot_loss": zero, "oai_fot_enabled": zero})

        idx = self._selected_indices(int(hidden.shape[1]), hidden.device)
        if idx.numel() < 2:
            return EmbeddingFoTOutput({"oai_fot_loss": zero, "oai_fot_enabled": zero})

        nodes = self.node_norm(hidden[:, idx, :].float())
        targets = target_ids[:, idx].long()
        nll = per_token_nll[:, idx].detach().float()
        if target_byte_lengths is not None:
            byte_lengths = target_byte_lengths[:, idx].detach().float().clamp_min(1.0)
        else:
            byte_lengths = torch.ones_like(nll)
        batch, n_nodes, _ = nodes.shape
        topology = self._forest_topology(int(n_nodes), hidden.device)
        num_trees = int(topology["num_trees"].item())
        tree_ids = topology["tree_ids"].long()
        depth_ids = topology["depth_ids"].long()
        branch_ids = topology["branch_ids"].long()
        parent_nodes = topology["parent_nodes"].long()
        child_nodes = topology["child_nodes"].long()
        edge_actions = topology["edge_actions"].long()
        temperature = max(float(cfg.temperature) * max(float(temperature_multiplier), 1.0e-4), 1.0e-4)
        ucb_exploration = max(float(cfg.ucb_exploration) * max(float(ucb_multiplier), 0.0), 0.0)

        activation_logits = self.activation_head(nodes).squeeze(-1).clamp(-30.0, 30.0)
        values = self.value_head(nodes).squeeze(-1).float()
        flow_values = self.flow_head(nodes).squeeze(-1).float()

        with torch.no_grad():
            norm_nodes = F.normalize(nodes.detach(), dim=-1)
            first_per_tree = torch.tensor(
                [int(torch.nonzero(tree_ids == t, as_tuple=False).flatten()[0].item()) for t in range(num_trees)],
                device=hidden.device,
                dtype=torch.long,
            )
            roots = norm_nodes[:, first_per_tree, :]
            root_for_node = roots[:, tree_ids, :]
            novelty = (1.0 - (norm_nodes * root_for_node).sum(dim=-1)).clamp(0.0, 2.0)
            depth_bonus = depth_ids.to(nodes.dtype).view(1, -1) / max(float(depth_ids.max().item() + 1), 1.0)
            byte_nll = nll / byte_lengths
            target_score = -byte_nll + float(cfg.reward_graph_weight) * novelty + 0.05 * depth_bonus
            tree_target_score = self._tree_reduce_mean(target_score, tree_ids, num_trees)
            activation_target = torch.softmax(tree_target_score / temperature, dim=-1)

        tree_activation_logits = self._tree_reduce_mean(activation_logits, tree_ids, num_trees)
        tree_log_probs = F.log_softmax(tree_activation_logits, dim=-1)
        sparse_loss = F.kl_div(tree_log_probs, activation_target, reduction="batchmean")
        activation_probs = torch.softmax(tree_activation_logits, dim=-1)
        activation_entropy = -(activation_probs * activation_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        activation_entropy_norm = activation_entropy / math.log(max(2, num_trees))
        topk = max(1, min(int(cfg.topk_trees), num_trees))
        active_mass = torch.topk(activation_probs, k=topk, dim=-1).values.sum(dim=-1).mean()
        active_tree_count = torch.exp(activation_entropy).detach()

        if child_nodes.numel() > 0:
            parent_hidden = nodes[:, parent_nodes, :]
            child_hidden = nodes[:, child_nodes, :]
            f_logits = self.forward_policy(parent_hidden).float().clamp(-30.0, 30.0)
            b_logits = self.backward_policy(child_hidden).float().clamp(-30.0, 30.0)
            actions = torch.remainder(edge_actions, int(cfg.branching)).view(1, -1).expand(batch, -1)
            log_pf = F.log_softmax(f_logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            log_pb = F.log_softmax(b_logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            visits = torch.arange(1, child_nodes.numel() + 1, device=hidden.device, dtype=torch.float32).view(1, -1)
            parent_visits = (depth_ids[parent_nodes].float().view(1, -1) + 1.0) * float(cfg.branching)
            child_visits = (branch_ids[child_nodes].float().view(1, -1) + 1.0).clamp_min(1.0)
            ucb_target_value = values[:, child_nodes].detach() + ucb_exploration * torch.sqrt(
                torch.log(parent_visits + 1.0) / child_visits.clamp_min(1.0)
            )
            ucb_weights = torch.softmax(ucb_target_value / temperature, dim=-1)
            transition_nll = F.cross_entropy(
                f_logits.reshape(-1, int(cfg.branching)),
                actions.reshape(-1),
                reduction="none",
            ).reshape(batch, -1)
            ucb_loss = (transition_nll * ucb_weights).sum(dim=-1).mean()

            correction = self.correction_head(parent_hidden).float()
            desired_delta = (child_hidden - parent_hidden).detach()
            correction_cosine = F.cosine_similarity(correction, desired_delta, dim=-1)
            correction_direction_loss = (1.0 - correction_cosine).mean()
            corrected_nodes = parent_hidden + float(cfg.correction_scale) * correction
            corrected_value = self.value_head(corrected_nodes).squeeze(-1).float()
            value_lift = corrected_value - values[:, parent_nodes].detach()
            correction_lift_loss = F.relu(0.01 - value_lift).mean()
            correction_loss = correction_direction_loss + 0.25 * correction_lift_loss

            raw_byte_nll = (nll / byte_lengths).mean(dim=1)
            corrected_byte_nll = raw_byte_nll
            bpb_delta_reward = hidden.new_zeros(batch, dtype=torch.float32)
            corrected_ce = nll[:, :-1].detach()
            if lm_head_weight is not None and str(cfg.reward_mode).strip().lower() in {"bpb_delta", "ce_delta", "byte_delta"}:
                logits_proj = F.linear(corrected_nodes.reshape(-1, corrected_nodes.shape[-1]), lm_head_weight.float())
                softcap = max(float(logit_softcap), 1.0e-4)
                logits_proj = softcap * torch.tanh(logits_proj / softcap)
                corrected_ce = F.cross_entropy(
                    logits_proj.float(),
                    targets[:, child_nodes].reshape(-1),
                    reduction="none",
                ).view(batch, -1)
                selected_bytes = byte_lengths[:, child_nodes].clamp_min(1.0)
                raw_local_byte_nll = (nll[:, child_nodes].detach() / selected_bytes).mean(dim=1)
                corrected_byte_nll = (corrected_ce / selected_bytes).mean(dim=1)
                bpb_delta_reward = (raw_local_byte_nll - corrected_byte_nll).clamp(-5.0, 5.0)
                reward = torch.exp(
                    -corrected_byte_nll.detach()
                    + float(cfg.bpb_delta_weight) * bpb_delta_reward.detach().clamp(-2.0, 2.0)
                )
            else:
                reward = torch.exp(-raw_byte_nll).clamp_min(1e-8)
            if advanced_signal is not None:
                bonus = torch.as_tensor(advanced_signal, device=hidden.device, dtype=torch.float32)
                while bonus.ndim > 1:
                    bonus = bonus.mean(dim=-1)
                if bonus.ndim == 0:
                    bonus = bonus.expand_as(reward)
                reward = reward * torch.exp(float(cfg.reward_advanced_bonus) * bonus[: reward.shape[0]].detach().clamp(-5.0, 5.0))
            complexity = (n_nodes / max(float(cfg.max_positions), 1.0)) + active_tree_count.float() / max(float(num_trees), 1.0)
            reward = (
                reward
                * torch.exp(float(cfg.reward_consensus_weight) * active_mass.detach().clamp(0.0, 1.0))
                * torch.exp(-float(cfg.reward_complexity_weight) * complexity.detach())
            ).clamp_min(max(float(cfg.reward_floor), 1.0e-8))
            tb_residual = self.log_z.float() + (log_pf - log_pb).mean(dim=1) - reward.log()
            tb_loss = tb_residual.square().mean()
            if child_nodes.numel() > 2:
                prefix = torch.cumsum(log_pf - log_pb, dim=1)
                denom = torch.arange(1, child_nodes.numel() + 1, device=hidden.device, dtype=torch.float32).view(1, -1)
                prefix_mean_byte_nll = torch.cumsum(
                    nll[:, child_nodes] / byte_lengths[:, child_nodes].clamp_min(1.0),
                    dim=1,
                ) / denom.clamp_min(1.0)
                prefix_reward = torch.exp(-prefix_mean_byte_nll).clamp_min(max(float(cfg.reward_floor), 1.0e-8))
                prefix_flow = flow_values[:, child_nodes]
                subtb_residual = prefix_flow + prefix - prefix_reward.log()
                subtb_loss = subtb_residual.square().mean()
            else:
                subtb_loss = zero.float()
        else:
            log_pf = log_pb = torch.empty(batch, 0, device=hidden.device)
            ucb_loss = zero.float()
            correction_loss = zero.float()
            correction_cosine = zero.float()
            value_lift = zero.float()
            tb_loss = zero.float()
            subtb_loss = zero.float()
            tb_residual = zero.float().expand(batch)
            reward = torch.exp(-(nll / byte_lengths).mean(dim=1)).clamp_min(max(float(cfg.reward_floor), 1.0e-8))
            complexity = zero.float()
            bpb_delta_reward = zero.float().expand(batch)
            corrected_ce = nll.detach()
            corrected_byte_nll = (nll / byte_lengths).mean(dim=1)

        leaf_idx = topology["leaf_indices"].long()
        leaf_nodes = nodes[:, leaf_idx, :]
        leaf_targets = torch.remainder(targets[:, leaf_idx], int(cfg.consensus_buckets))
        leaf_logits = self.consensus_head(leaf_nodes).float().clamp(-30.0, 30.0)
        tree_weights = torch.softmax(tree_activation_logits, dim=-1).unsqueeze(-1)
        consensus_logits = (tree_weights * leaf_logits).sum(dim=1)
        final_target = torch.remainder(target_ids[:, idx[-1]], int(cfg.consensus_buckets))
        consensus_loss = F.cross_entropy(consensus_logits, final_target)
        consensus_probs = torch.softmax(consensus_logits, dim=-1)
        consensus_top2 = torch.topk(consensus_logits, k=min(2, int(cfg.consensus_buckets)), dim=-1).values
        consensus_margin = (
            consensus_top2[:, 0] - consensus_top2[:, -1] if consensus_top2.shape[-1] > 1 else consensus_top2[:, 0]
        ).mean()
        consensus_entropy = -(consensus_probs * consensus_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        leaf_pred = leaf_logits.argmax(dim=-1)
        tree_agreement = (leaf_pred == final_target.view(-1, 1)).float().mean()
        leaf_supervision_loss = F.cross_entropy(
            leaf_logits.reshape(-1, int(cfg.consensus_buckets)),
            leaf_targets.reshape(-1),
        )
        consensus_loss = 0.75 * consensus_loss + 0.25 * leaf_supervision_loss

        tree_repr = self._tree_reduce_mean(F.normalize(nodes, dim=-1), tree_ids, num_trees)
        if num_trees > 1:
            sim = torch.matmul(tree_repr, tree_repr.transpose(1, 2))
            mask = ~torch.eye(num_trees, device=hidden.device, dtype=torch.bool).unsqueeze(0)
            tree_diversity = (1.0 - sim.masked_select(mask).view(batch, -1).mean(dim=-1)).mean().clamp(0.0, 2.0)
        else:
            tree_diversity = zero.float()
        complexity_loss = (
            F.relu(active_tree_count.float() / max(float(num_trees), 1.0) - 0.85).square()
            + F.relu(0.10 - tree_diversity).square()
        )

        total = (
            float(cfg.sparse_weight) * max(float(sparse_multiplier), 0.0) * sparse_loss
            + float(cfg.ucb_weight) * ucb_loss
            + float(cfg.correction_weight) * correction_loss
            + float(cfg.consensus_weight) * consensus_loss
            + float(cfg.tb_weight) * tb_loss
            + float(cfg.subtb_weight) * subtb_loss
            + float(cfg.complexity_weight) * complexity_loss
        )
        total = torch.nan_to_num(total, nan=0.0, posinf=1.0e4, neginf=0.0)

        return EmbeddingFoTOutput(
            {
                "oai_fot_loss": total,
                "oai_fot_sparse_activation_loss": sparse_loss.detach(),
                "oai_fot_ucb_loss": ucb_loss.detach(),
                "oai_fot_self_correction_loss": correction_loss.detach(),
                "oai_fot_consensus_loss": consensus_loss.detach(),
                "oai_fot_tb_loss": tb_loss.detach(),
                "oai_fot_subtb_loss": subtb_loss.detach(),
                "oai_fot_complexity_loss": complexity_loss.detach(),
                "oai_fot_activation_entropy": activation_entropy_norm.detach(),
                "oai_fot_active_tree_count": active_tree_count.detach(),
                "oai_fot_active_mass_topk": active_mass.detach(),
                "oai_fot_tree_diversity": tree_diversity.detach(),
                "oai_fot_value_mean": values.detach().mean(),
                "oai_fot_reward_mean": reward.detach().mean(),
                "oai_fot_reward_bpb_delta": bpb_delta_reward.detach().mean(),
                "oai_fot_reward_raw_byte_nll": raw_byte_nll.detach().mean() if "raw_byte_nll" in locals() else zero.detach(),
                "oai_fot_reward_corrected_byte_nll": (
                    corrected_byte_nll.detach().mean() if torch.is_tensor(corrected_byte_nll) else zero.detach()
                ),
                "oai_fot_corrected_ce": corrected_ce.detach().mean() if torch.is_tensor(corrected_ce) else zero.detach(),
                "oai_fot_tb_residual": tb_residual.detach().abs().mean(),
                "oai_fot_correction_cosine": correction_cosine.detach().mean(),
                "oai_fot_correction_bpb_proxy_lift": value_lift.detach().mean() if torch.is_tensor(value_lift) else zero.detach(),
                "oai_fot_consensus_margin": consensus_margin.detach(),
                "oai_fot_consensus_entropy": (
                    consensus_entropy / math.log(max(2, int(cfg.consensus_buckets)))
                ).detach(),
                "oai_fot_consensus_tree_agreement": tree_agreement.detach(),
                "oai_fot_log_z": self.log_z.detach(),
                "oai_fot_num_trees": hidden.new_tensor(float(num_trees)).detach(),
                "oai_fot_node_count": hidden.new_tensor(float(n_nodes)).detach(),
                "oai_fot_edge_count": hidden.new_tensor(float(child_nodes.numel())).detach(),
                "oai_fot_max_depth_observed": depth_ids.max().float().detach(),
                "oai_fot_topk_trees": hidden.new_tensor(float(topk)).detach(),
                "oai_fot_enabled": hidden.new_tensor(1.0).detach(),
            }
        )

    @torch.no_grad()
    def trace_payload(self, hidden: Tensor, target_ids: Tensor, per_token_nll: Tensor) -> dict[str, object]:
        """Return a compact JSON-serializable forest trace for inference/reporting."""
        idx = self._selected_indices(int(hidden.shape[1]), hidden.device)
        nodes = self.node_norm(hidden[:, idx, :].float())
        n_nodes = int(nodes.shape[1])
        topology = self._forest_topology(n_nodes, hidden.device)
        num_trees = int(topology["num_trees"].item())
        tree_ids = topology["tree_ids"].long()
        depth_ids = topology["depth_ids"].long()
        branch_ids = topology["branch_ids"].long()
        parent_nodes = topology["parent_nodes"].long()
        child_nodes = topology["child_nodes"].long()
        leaf_indices = topology["leaf_indices"].long()
        activation = self.activation_head(nodes).squeeze(-1).float()[0]
        value = self.value_head(nodes).squeeze(-1).float()[0]
        tree_activation = self._tree_reduce_mean(activation.view(1, -1), tree_ids, num_trees)[0]
        tree_probs = torch.softmax(tree_activation, dim=-1)
        leaf_nodes = nodes[:, leaf_indices, :]
        leaf_logits = self.consensus_head(leaf_nodes).float()[0]
        consensus_logits = (tree_probs.view(-1, 1) * leaf_logits).sum(dim=0)
        consensus_probs = torch.softmax(consensus_logits, dim=-1)
        consensus_bucket = int(consensus_probs.argmax().detach().cpu().item())

        edges: list[dict[str, object]] = [{"source": "forest_root", "target": int(i), "tree_id": int(i), "kind": "tree_seed"} for i in range(num_trees)]
        for parent, child in zip(parent_nodes.tolist(), child_nodes.tolist(), strict=False):
            edges.append(
                {
                    "source": int(parent),
                    "target": int(child),
                    "tree_id": int(tree_ids[child].item()),
                    "branch_id": int(branch_ids[child].item()),
                    "depth": int(depth_ids[child].item()),
                    "kind": "tree_expansion",
                }
            )
            edges.append(
                {
                    "source": int(parent),
                    "target": int(child),
                    "tree_id": int(tree_ids[child].item()),
                    "branch_id": int(branch_ids[child].item()),
                    "depth": int(depth_ids[child].item()),
                    "kind": "self_correction",
                }
            )
        for tree_id, leaf in enumerate(leaf_indices.tolist()):
            edges.append(
                {
                    "source": int(leaf),
                    "target": "consensus",
                    "tree_id": int(tree_id),
                    "kind": "consensus_vote",
                    "tree_probability": float(tree_probs[tree_id].detach().cpu().item()),
                }
            )
        node_payload: list[dict[str, object]] = [
            {
                "id": "forest_root",
                "type": "forest_root",
                "tree_id": None,
                "depth": -1,
                "branch_id": None,
                "position": None,
                "nll": 0.0,
                "activation": 0.0,
                "value": 0.0,
            }
        ]
        node_payload.extend(
            [
                {
                    "id": int(i),
                    "type": "thought_state",
                    "tree_id": int(tree_ids[i].item()),
                    "depth": int(depth_ids[i].item()),
                    "branch_id": int(branch_ids[i].item()),
                    "position": int(idx[i].item()),
                    "target_id": int(target_ids[0, idx[i]].item()),
                    "nll": float(per_token_nll[0, idx[i]].detach().float().item()),
                    "activation": float(activation[i].detach().cpu().item()),
                    "value": float(value[i].detach().cpu().item()),
                    "is_leaf": bool(i in set(leaf_indices.tolist())),
                }
                for i in range(n_nodes)
            ]
        )
        node_payload.append(
            {
                "id": "consensus",
                "type": "forest_consensus",
                "tree_id": None,
                "depth": int(depth_ids.max().detach().cpu().item()) + 1 if n_nodes else 0,
                "branch_id": None,
                "position": None,
                "nll": 0.0,
                "activation": 0.0,
                "value": float(consensus_probs.max().detach().cpu().item()),
                "consensus_bucket": consensus_bucket,
            }
        )
        return {
            "schema": "toricgt.embedding_forest_of_thought.trace.v1",
            "reference": "external/Forest-of-Thought",
            "num_trees": num_trees,
            "node_count": n_nodes,
            "edge_count": len(edges),
            "max_depth": int(depth_ids.max().detach().cpu().item()) if n_nodes else 0,
            "branching": int(self.config.branching),
            "consensus": {
                "bucket": consensus_bucket,
                "confidence": float(consensus_probs.max().detach().cpu().item()),
                "entropy": float((-(consensus_probs * consensus_probs.clamp_min(1e-8).log()).sum()).detach().cpu().item()),
            },
            "tree_summaries": [
                {
                    "tree_id": int(tree_id),
                    "activation_probability": float(tree_probs[tree_id].detach().cpu().item()),
                    "leaf_node": int(leaf_indices[tree_id].detach().cpu().item()),
                    "leaf_value": float(value[leaf_indices[tree_id]].detach().cpu().item()),
                }
                for tree_id in range(num_trees)
            ],
            "nodes": node_payload,
            "edges": edges,
        }
