# Embedding-Space Forest-of-Thought Adaptation Plan

Status: implemented and active in `tg-fot-bpb119-1k-20260618T223154Z`
Scope: ToricGT OAI Parameter-Golf baseline adaptation only. Do not migrate work into TropicalGT.
Primary objective: keep FineWeb/OAI BPB as the score-defining objective while adding a training-only embedding-space Forest-of-Thought (FoT) controller that is compatible with TokenGT graphification, GraphCG, GFlowNets, toric/tropical/topological/category losses, test-time scaling, and the existing adaptive 25+10 campaign loop.

## 1. Sources Reviewed

### Local FoT Repository

Repository cloned at `external/Forest-of-Thought` from `https://github.com/amelie-iska/Forest-of-Thought.git`.

Important files:

- `README.md`: states the ICML 2025 FoT objective and arguments: number of trees, correction toggle, evaluation samples, selection samples, and benchmark scripts for Game24/GSM8K/MATH500/AIME.
- `run.py`: wrapper for Game24 FoT experiments; dispatches to `forest_solve`.
- `methods/bfs.py`: Game24 forest solver. It creates multiple shuffled-input trees, expands candidate steps, scores them, selects high-value paths, and optionally corrects malformed arithmetic steps.
- `run_with_mcf.py`: Monte-Carlo-Forest implementation for math benchmarks. It builds multiple reasoning trees, scores answers, updates UCB values, inserts weak/bad answers for exploration contrast, and uses majority/score/scaling stopping.
- `run_with_mcf_stop_noearly.py`: richer FoT script with base modes `cot`, `tot`, and `mcts`; includes sparse activation signals, consensus-guided final selection, and dynamic self-correction.
- `methods/mcts/mcts.py` and `methods/mcts/base.py`: MCTS state mechanics: `treeNode`, visits, value, reflection, expansion, rollout, backpropagation, and best-value extraction.
- `methods/tot/bfs.py`: ToT-style expansion/evaluation/selection over coherent “thought” units.
- `cgdm/cgdm.py`: consensus-guided decision helper using an expert-choice model over candidate answers.

Transferable FoT primitives:

- A forest is a set of multiple independent reasoning trees.
- Each tree can be initialized by a different input view, exemplar, prompt, or latent perturbation.
- Nodes store a partial reasoning state, parent, children, visit count, scalar value/reward, reflection/correction metadata, and terminal-answer metadata.
- Tree expansion is value-guided, typically by UCB or greedy top-value selection.
- Sparse activation chooses which trees/paths remain compute-active.
- Dynamic self-correction asks for feedback on weak states and generates corrected children.
- Consensus-guided decision-making combines multiple tree outputs through majority, score, or expert/CGED-style selection.

### Paper Review

Primary paper:

- Bi et al., “Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning,” arXiv:2412.09078 / ICML 2025.

Core conclusions to preserve:

- FoT improves on single-pass CoT/ToT because it revisits flawed paths with multiple reasoning trees rather than committing to one path.
- Sparse activation is essential because naively activating all trees wastes compute; activation should track relevance, confidence, diversity, and expected value.
- Dynamic self-correction is not just a prompt trick. It prevents early reasoning mistakes from compounding through the tree.
- Consensus-guided decision-making is better than random or score-only leaf selection, especially as the number of active subtrees grows.
- Accuracy improves with activated subtree count, but with diminishing returns; this should become an adaptive compute-budget controller in ToricGT.

Related methods and why they matter:

- Tree of Thoughts (ToT), arXiv:2305.10601: expands coherent thought units and uses self-evaluation/search. ToricGT should treat local hidden spans or graph nodes as thought units.
- Graph of Thoughts (GoT), arXiv:2308.09687: supports arbitrary thought DAGs with aggregation, feedback loops, and merging. ToricGT already has graph-of-thought visualizations; FoT should sit above GoT as a forest of graph/DAG trajectories.
- Self-Consistency, arXiv:2203.11171: samples diverse reasoning paths and marginalizes over answers. ToricGT’s consensus head should compute a differentiable analogue over embedding-space leaves.
- Least-to-Most Prompting, arXiv:2205.10625: decomposes complex problems into subproblems. FoT tree roots can be seeded by low-complexity-to-high-complexity subproblem embeddings.
- Program of Thoughts, arXiv:2211.12588: delegates computation to symbolic programs. In ToricGT, CAS-backed toric/Koszul/BGG sidecars provide the analogous “external computation” validation.
- MCTS-DPO, arXiv:2405.00451: uses MCTS to create step-level preference data. ToricGT should train value/correction heads with step-level BPB improvement and advanced verifier signals.
- Coconut latent reasoning, arXiv:2412.06769: continuous hidden states can represent multiple possible next steps and show BFS-like behavior. This supports implementing FoT in embedding space rather than textual prompt space.
- Continuous GFlowNets, arXiv:2301.12594, and GFlowNet foundations, arXiv:2111.09266: provide the reward-proportional sampling objective for continuous/hybrid reasoning trajectories.

Already-cloned reference:

- `external/gflownet` provides a full GFlowNet framework with continuous environments and trajectory-balance losses. For the OAI baseline, we should not import the Hydra-heavy training framework directly. Instead, we should implement a compact self-contained FoT trajectory-balance head in `train_gpt.py`/`src/toricgt`, using the official repo as a correctness reference for flow balance concepts.

## 2. Design Principles

1. BPB remains primary. FoT losses must be routed through the existing auxiliary-gradient controller and must be reduced or projected when they conflict with FineWeb CE/BPB.
2. FoT is training-only by default. It must not increase the serialized Parameter-Golf artifact unless an explicit export path is later chosen and size-tested.
3. Preserve all existing techniques. FoT must coexist with first-class TokenGT graphification, graph-output flattening, graph LM stream, OAI GFlowNet, OAI MTP, ToricGT sidecar, GraphCG, tropical/toric geometry, vector-bundle/sheaf, Toric BGG, Koszul persistence, combinatorial toric commutative algebra, derived signatures, CAS audits, and W&B reporting.
4. Use embedding-space structures. Textual FoT prompts become latent roots, nodes, corrections, activations, and consensus decisions over hidden states.
5. Use graph structure where it helps. FineWeb is graphified internally but flattened for BPB scoring. Graph-structured rows remain graph-in/graph-out. FoT should operate on both by seeing hidden-state trajectories plus token/graph position metadata.
6. Keep the OAI baseline self-contained. New code may live in `src/toricgt`, but `amelie-iska/parameter-golf/train_gpt.py` should remain the training entry point.

## 3. Embedding FoT Objects

### Thought Node

For a batch hidden state `H in R[B, T, D]`, construct a bounded forest over a selected subset of token positions.

```python
EmbeddingThoughtNode:
    tree_id: int
    node_id: int
    parent_id: int | None
    token_position: int
    hidden: Tensor[D]
    depth: int
    visit_count: float
    value: float
    reward: float
    activation_logit: float
    correction: Tensor[D]
    graphcg_signature: Tensor[R]
    topological_signature: Tensor[K]
    toric_signature: Tensor[K2]
```

### Forest

```python
EmbeddingFoTForest:
    roots: list[EmbeddingThoughtNode]
    nodes: list[EmbeddingThoughtNode]
    tree_edges: list[(parent_id, child_id)]
    merge_edges: list[(source_node_id, target_node_id)]
    active_tree_mask: Tensor[num_trees]
    consensus_clusters: Tensor[num_leaves]
```

Forest construction for training should be deterministic and bounded:

- Sample at most `OAI_FOT_MAX_POSITIONS` positions from the last FineWeb microbatch.
- Divide positions into `OAI_FOT_NUM_TREES` interleaved roots.
- Create tree edges from causal predecessor windows and nearest previous hidden states.
- Allow limited merge edges when two branches converge in hidden/GraphCG/PH signature space.
- For FineWeb scoring, maintain sequence order for flattening; FoT does not change target labels directly.

## 4. Training Objectives

### Sparse Tree Activation

FoT activation should select trees/nodes with strong expected BPB usefulness:

```python
activation_target = softmax(
    -local_nll
    + novelty_bonus
    + graphcg_stability
    + toric_fan_margin
    + memory_sheaf_agreement
)
L_sparse = KL(activation_target || softmax(activation_logits)) + entropy_budget_penalty
```

Metrics:

- `oai_fot/sparse_activation_loss`
- `oai_fot/activation_entropy`
- `oai_fot/active_tree_count`
- `oai_fot/tree_diversity`

### UCB / Value-Guided Expansion

Use a differentiable UCB target:

```python
ucb = value + c * sqrt(log(parent_visits + 1) / (child_visits + eps))
L_ucb = CE(policy_logits, softmax(ucb / temperature))
```

Metrics:

- `oai_fot/ucb_loss`
- `oai_fot/value_mean`
- `oai_fot/exploration_bonus_mean`

### Dynamic Self-Correction

Predict a correction vector that moves a weak node toward a better local future state:

```python
corrected = hidden + correction_scale * correction_head(hidden)
teacher_delta = low_nll_future_hidden.detach() - hidden.detach()
L_correct = 1 - cosine(correction, teacher_delta)
L_correct += hinge(local_ce_after_correction - local_ce_before)
```

This is the embedding-space analogue of FoT reflection/refinement. It should be gated by uncertainty and should not overwrite the main forward pass unless the BPB-safe calibration path shows improvement.

Metrics:

- `oai_fot/self_correction_loss`
- `oai_fot/correction_cosine`
- `oai_fot/correction_norm`
- `oai_fot/correction_bpb_proxy_lift`

### Consensus-Guided Decision

Leaves vote through a consensus head. For BPB training, the “answer” is the next-token distribution or compact class bucket, not a textual final answer:

```python
leaf_logits = consensus_head(leaf_hidden)
tree_weights = softmax(tree_value + activation_logit - complexity_penalty)
consensus_logits = weighted_sum(tree_weights, leaf_logits)
L_consensus = CE(consensus_logits, target_token_or_bucket)
L_consistency = variance_penalty among high-value trees that agree on target bucket
```

Metrics:

- `oai_fot/consensus_loss`
- `oai_fot/consensus_margin`
- `oai_fot/consensus_entropy`
- `oai_fot/consensus_tree_agreement`

### FoT GFlowNet Balance

Train forest construction as a reward-proportional sampler:

```python
reward = exp(-mean_nll) * exp(-complexity_cost)
reward *= exp(+advanced_metric_agreement_bonus)
tb = logZ + sum(log_pf) - sum(log_pb) - log(reward)
L_tb = mean(tb^2)
```

Subtrajectory terms should be enabled later:

```python
L_subtb = mean_{i<j} (F(s_i) + log P_F(i:j) - F(s_j) - log P_B(j:i))^2
```

Metrics:

- `oai_fot/tb_loss`
- `oai_fot/tb_residual`
- `oai_fot/subtb_loss`
- `oai_fot/reward_mean`
- `oai_fot/log_z`

### Total FoT Loss

```python
L_fot =
    w_sparse * L_sparse
  + w_ucb * L_ucb
  + w_correct * L_correct
  + w_consensus * L_consensus
  + w_tb * L_tb
  + w_complexity * L_complexity
```

This total is multiplied by `OAI_FOT_LOSS_WEIGHT` and routed as an auxiliary gradient named `oai_fot`.

## 5. Integration Points

### `src/toricgt/embedding_forest_of_thought.py`

Implement:

- `EmbeddingFoTConfig`
- `EmbeddingForestOfThoughtHead`
- `EmbeddingFoTOutput`
- Optional `build_trace_payload(...)` for visualization/inference.

The module must:

- Use only PyTorch and standard library.
- Be deterministic for a fixed seed and input.
- Support CPU smoke tests.
- Avoid large all-pairs memory; limit selected positions.
- Return scalar losses and W&B-ready metrics.

### `amelie-iska/parameter-golf/train_gpt.py`

Add hyperparameters:

- `OAI_EMBEDDING_FOT`
- `OAI_FOT_LR`
- `OAI_FOT_EVERY`
- `OAI_FOT_LOSS_WEIGHT`
- `OAI_FOT_NUM_TREES`
- `OAI_FOT_MAX_DEPTH`
- `OAI_FOT_BRANCHING`
- `OAI_FOT_TOPK_TREES`
- `OAI_FOT_MAX_POSITIONS`
- `OAI_FOT_SPARSE_WEIGHT`
- `OAI_FOT_UCB_WEIGHT`
- `OAI_FOT_CORRECTION_WEIGHT`
- `OAI_FOT_CONSENSUS_WEIGHT`
- `OAI_FOT_TB_WEIGHT`
- `OAI_FOT_SUBTB_WEIGHT`
- `OAI_FOT_COMPLEXITY_WEIGHT`
- `OAI_FOT_REWARD_ADVANCED_BONUS`

Add:

- Head construction and optimizer.
- Checkpoint load/save entries.
- Train-loop block after OAI GFlowNet and MTP or before sidecar.
- W&B logging and train-log brief.
- Aux-gradient routing named `oai_fot`.

### Adaptive Campaign Controller

Update:

- `scripts/adaptive_bpb_annealing.py`
- `scripts/run_oai_sidecar_bpb_campaign.py`
- `scripts/analyze_toricgt_sidecar_wandb_metrics.py`
- `src/toricgt/wandb_organization.py`

Add FoT family:

- `OAI_FOT_LOSS_WEIGHT`
- `OAI_FOT_CORRECTION_WEIGHT`
- `OAI_FOT_CONSENSUS_WEIGHT`
- `OAI_FOT_TB_WEIGHT`
- `OAI_FOT_NUM_TREES`
- `OAI_FOT_TOPK_TREES`

Decision logic:

- If train BPB falls too slowly and FoT gradients align with primary CE, increase FoT and consensus modestly.
- If FoT TB residual explodes or aux conflict is negative, reduce FoT total and correction weight.
- If activation entropy collapses, increase sparse entropy/active trees or reduce UCB sharpness.
- If consensus margin is high but BPB does not improve, lower consensus and raise direct BPB/native MTP.
- If tree diversity is low, increase `num_trees` or branch diversity, but respect runtime and memory.

### Analysis and Visualization

Add FoT analysis outputs:

- FoT metric report section in each 1K-step analysis report.
- HTML visualization for a compact forest trace: multiple trees, active paths, correction edges, merge edges, consensus leaf clusters, and BPB-colored nodes.
- W&B metrics descriptions for each FoT metric.

The full heavy visualizations remain optional and should not run inside every training step. The campaign analysis can render them every gate using the latest checkpoint/log artifacts.

## 6. New Campaign Requirements

Once implementation and tests pass:

- Stop the existing non-FoT BPB campaign.
- Start a new tmux session with run 001.
- Target: `val_bpb < 1.19` by `1000` steps.
- Iterations: 25 runs, then meta-analysis, then 10 follow-up runs.
- Use all existing advanced losses/metrics plus FoT.
- Keep `DATA_PATH=/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data` only as the dataset source path. Project code and outputs remain ToricGT.
- Use the existing OAI baseline adaptation with TokenGT graphification and graph-output flattening for FineWeb BPB.
- Keep score-first test/inference-time scaling enabled as configured by the campaign profiles.

## 7. Implementation Checklist

- [x] Clone `Forest-of-Thought` into `external/Forest-of-Thought`.
- [x] Inspect FoT repository implementation and identify transferable primitives.
- [x] Review FoT paper and related reasoning/GFlowNet/latent-reasoning sources.
- [x] Write this implementation plan.
- [x] Implement `src/toricgt/embedding_forest_of_thought.py`.
- [x] Add OAI FoT hyperparameters and training integration.
- [x] Add FoT optimizer, checkpoint, logging, W&B, and aux-gradient routing.
- [x] Add FoT adaptive-controller rules.
- [x] Add FoT campaign profile defaults for 1K-step BPB gates.
- [x] Add FoT metric descriptions and report sections.
- [x] Add FoT visualization renderer or analysis artifact hook.
- [x] Run Python compile checks.
- [x] Run CPU FoT head smoke test.
- [x] Defer a second live `train_gpt.py` smoke test while the current GPU training loop is active; the existing loop has already picked up FoT in run 006, which is the live integration smoke.
- [x] Start new FoT 25+10 training campaign only after tests pass.

## 8. Initial Conjectures for ToricGT

1. FoT should improve early BPB only if it is used as a low-weight routing and representation-shaping signal, not as a high-weight detached reasoning objective. Early BPB is dominated by lexical modeling; FoT should help by stabilizing hidden search and memory, not by forcing expensive explicit reasoning on every token.
2. Consensus should be bucket-level for FineWeb. Full-token consensus over 1024 classes is useful but expensive; a token-class/boundary/byte-length consensus target can regularize graphified output without competing with the main LM head.
3. Dynamic self-correction should be uncertainty-gated. Applying correction everywhere will likely damage BPB; applying it to high-NLL/high-entropy regions should improve difficult local predictions while preserving easy-token flow.
4. FoT and GraphCG should cooperate: trees should diversify along GraphCG axes, while consensus collapses only after topological/toric/memory agreement is high enough.
5. FoT’s sparse activation can become an adaptive compute signal for inference-time scaling. The model should learn when to spend additional latent forest compute, but this must remain score-first and optional for the competition artifact.
