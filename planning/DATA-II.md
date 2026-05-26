# DATA-II: OpenAI Benchmarks, GraphWalks, And Higher-Order Simplicial Augmentation

Author: Amelie Schreiber

## Purpose

This plan extends the ToricGT data program in two layers.

First, the OpenAI datasets below are integrated into the curation manifest now:

- `openai/frontierscience`
- `openai/healthbench`
- `openai/healthbench-professional`
- `openai/graphwalks`

Second, the SauravMaheshkar/Cornell temporal hypergraph datasets are evaluated
as candidate higher-order graph augmentation sources, but are not enabled until
explicit approval.  They are useful for simplicial closure, clique-complex
reasoning, temporal hyperedge prediction, and persistent-homology tasks, but
their Hugging Face cards currently report `license:unknown`, so they should not
enter the default public training mixture without license review.

## OpenAI Dataset Integration

| Dataset | Observed splits | License | ToricGT policy | Role |
|---|---:|---|---|---|
| `openai/frontierscience` | `test`, 160 rows | Apache-2.0 | eval-only, forced `test` | expert-level scientific tasks across subjects |
| `openai/healthbench` | `test`, 5,000 rows | MIT | eval-only, forced `test` | rubric-based health response evaluation |
| `openai/healthbench-professional` | `test`, 525 rows | MIT | eval-only, forced `test` | physician-response professional health evaluation |
| `openai/graphwalks` | `train`, 1,150 rows | MIT | forced `test` for inference-time scaling | directed multi-hop graph operation reasoning |

The eval-only decision is intentional.  `frontierscience` is explicitly marked
on its dataset card as benchmark material that should not enter training
corpora.  HealthBench and HealthBench Professional are also evaluation datasets,
and they are medical-domain rubrics; using them as ordinary training examples
would contaminate evals and create a misleading health-capability signal.

`openai/graphwalks` is different structurally: the source split is named
`train`, and its examples contain explicit directed edge-list prompts plus
target answer-node sets.  Per the current contamination policy, it is still
forced into ToricGT's `test` split and used for test-time/inference-time
scaling rather than training.  The local implementation converts those prompts
into graph-native JSON with graph-node tokens, directed-edge tokens, an
operation node, and answer-node edges.

## OpenAI Normalization Details

### FrontierScience

Schema observed through `datasets`:

```text
problem, answer, subject, task_group_id
```

Normalization:

- `question = problem`
- `answer = answer`
- `reasoning = subject`
- `family_key = task_group_id`
- `split = test`

Use only for held-out scientific evaluation.

### HealthBench

Streaming schema observed:

```text
example_tags, ideal_completions_data, prompt, prompt_id, rubrics, canary
```

Normalization:

- `question = prompt messages`
- `answer/solution/reasoning = rubric criteria`
- `family_key = prompt_id`
- `split = test`

Use only for rubric-eval graph conversion and scorer development.  Do not train
the model to reproduce these rubrics as ordinary SFT targets unless a separate
contamination-safe medical-eval protocol is approved.

### HealthBench Professional

Schema observed:

```text
id, conversation, rubric_items, use_case, type, difficulty, specialty,
physician_response, canary_string
```

Normalization:

- `question = conversation messages`
- `answer = physician_response`
- `reasoning = rubric_items`
- `family_key = id`
- `split = test`

Use for professional health benchmark auditing only.

### GraphWalks

Schema observed:

```text
prompt, answer_nodes, prompt_chars, problem_type, date_added
```

Normalization:

- parse prompt lines matching `source -> target`;
- create one `graph_node` node per node id;
- create one `directed_edge` edge per edge-list row;
- create an `operation` node from the operation text;
- create `returns_answer_node` edges from the operation to each target answer
  node.

This gives ToricGT actual graph-token held-out evaluation examples instead of
treating the graph as only a string.

## Test-Time Scaling Protocol For New Datasets

All newly added external reasoning datasets are reserved for test-time or
inference-time scaling unless explicitly moved into a future approved training
mixture.  This includes the NVIDIA/Nemotron datasets added in the previous
cycle and the OpenAI datasets listed above.  The curation code therefore sets
`forced_split="test"` for these sources.

Run held-out scaling with:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src python scripts/test_time_scaling.py \
  --checkpoint checkpoints/toricgt_full_30m/toricgt_step_00002000.pt \
  --device cuda \
  --batch-size 2 \
  --batches 20 \
  --budgets 1 4 16 64 \
  --gflownet-horizon 4 \
  --output-json runs/test_time_scaling/openai_nvidia_eval.json
```

Metrics:

- `mean_mse`: average candidate graph reconstruction loss at a budget;
- `best_oracle_mse`: best candidate under an oracle verifier, used only as an
  upper-bound diagnostic;
- `mean_reward_proxy` and `best_reward_proxy`: `exp(-loss)` reward proxies;
- `policy_entropy`: GFlowNet action entropy;
- `action_diversity`: unique sampled GFlowNet action coverage;
- `mean_trajectory_logprob`: average sampled trajectory log-probability.

This script does not interrupt training.  It loads a saved checkpoint in a
separate process and can be run on CPU or CUDA.  By default it uses the
OpenAI/NVIDIA held-out dataset group and auto-curates
`data/test_time_scaling/new_external/test.parquet` if the file is absent.
GFlowNet rollouts and Monte Carlo dropout are enabled by default.  If GPU memory
is tight, run it on CPU or wait for the next checkpoint; do not stop the active
training job just to evaluate these held-out datasets.

## Candidate Simplicial Datasets

The three requested SauravMaheshkar datasets are wrappers around temporal
hypergraph datasets associated with Benson, Abebe, Schaub, Jadbabaie, and
Kleinberg's *Simplicial Closure and higher-order link prediction*
(`arXiv:1802.06916`).  The paper studies temporal higher-order interactions,
models events as hyperedges/simplices, and frames higher-order link prediction
as predicting future group interactions rather than only pairwise edges.  A key
modeling lesson for ToricGT is that local higher-order structure, not merely
long-range pairwise graph connectivity, is central for these prediction tasks.

| Dataset | Configs | Observed transductive rows | Helpfulness | Viability |
|---|---|---:|---|---|
| `SauravMaheshkar/tags-math-sx` | `raw`, `transductive`, `inductive` | train 575,441 / valid 123,309 / test 123,309 | High: math StackExchange tag simplices are directly relevant to mathematical concept closure and clique prediction. | Good technically, blocked by license review. |
| `SauravMaheshkar/tags-ask-ubuntu` | `raw`, `transductive`, `inductive` | train 189,863 / valid 40,685 / test 40,685 | Medium-high: software-tag hyperedges help algorithmic taxonomy and concept co-occurrence. | Good technically, blocked by license review. |
| `SauravMaheshkar/NDC-substances-25` | `raw`, `transductive`, `inductive` | train 78,696 / valid 16,888 / test 16,821 | Medium: chemical/substance hyperedges are useful for non-textual higher-order closure, but less central to ToricGT's current reasoning focus. | Technically easy, but license and biomedical framing require caution. |

Observed row schema:

```text
hyperedge: int64
nodes: string-encoded list of node ids
timestamp: float64
```

## Recommendation

Do not add the SauravMaheshkar datasets to the default manifest yet.

Recommended approval path:

1. Resolve license and redistribution terms.
2. Add them behind an explicit `--dataset` selector rather than to the default
   public curation list.
3. Prefer `raw` for synthetic persistent-homology tasks and `inductive` for
   generalization tests.
4. Preserve the original temporal order; do not random-split raw hyperedges.
5. Use the published transductive/inductive splits only for benchmark
   reproduction, not for leakage-controlled mixed-corpus training.

## Simplicial Graph Conversion

For a hyperedge event

```text
e_t = {v_1, ..., v_k} at timestamp t
```

construct a typed graph with:

- one `simplex_event` node;
- one `vertex` node per participating node id;
- optional `simplex_face_j` nodes for all non-empty faces up to a configured
  dimension cap;
- `contains_vertex`, `has_face`, `face_of`, and `coface_of` edges;
- a `timestamp` scalar or positional bucket;
- optional clique-expansion pair edges marked as `shadow_pair`.

For large hyperedges, cap explicit face expansion at dimension 2 or 3 and store
higher-order membership as sparse incidence edges.  Full power-set expansion is
not acceptable for large tags or substance sets.

## Persistent-Homology Training Methods

These datasets can support new ToricGT objectives after approval.

### 1. Simplicial Closure Prediction

Given all hyperedges before time `t`, predict whether a candidate set
`{v_1, ..., v_k}` appears after `t`.

Targets:

- binary closure label;
- time-to-event bucket;
- minimal missing face;
- local density and tie-strength bins.

Losses:

- binary cross entropy for closure;
- ordinal loss for time bucket;
- contrastive loss between true future simplices and hard negative cliques.

### 2. Boundary-Operator Reconstruction

Construct sparse boundary matrices

```text
partial_k: C_k -> C_{k-1}
```

from simplex tokens.  Train the model to predict signed face incidence and to
respect `partial_{k-1} partial_k = 0`.

Losses:

- sparse incidence reconstruction;
- chain-complex consistency penalty;
- equivariance error under vertex relabeling.

### 3. Hodge-Laplacian Reasoning

For dimension `k`, use the combinatorial Hodge Laplacian

```text
L_k = partial_{k+1} partial_{k+1}^T + partial_k^T partial_k
```

as a supervision source.

Tasks:

- predict local harmonic/non-harmonic simplex labels;
- regress low-order spectral summaries;
- classify whether adding a hyperedge kills or creates a cycle.

### 4. Persistence Diagram And Betti-Curve Prediction

Use timestamp order or learned weights as a filtration.  Train on:

- Betti curves `beta_0(t), beta_1(t), beta_2(t)`;
- persistence-pair birth/death buckets;
- persistence image summaries;
- sliced-Wasserstein or heat-kernel distances between predicted and target
  persistence summaries.

This should be an auxiliary objective, not the main supervised objective,
because exact persistence can be expensive at corpus scale.

### 5. GFlowNet Complex Construction

Use a GFlowNet over embedding-space graph states to construct candidate
simplicial complexes.

Actions:

- add a vertex;
- add a candidate simplex;
- add all faces of a simplex;
- close a clique into a simplex;
- delete/penalize an inconsistent candidate;
- terminate.

Reward:

```text
R = exp(
  lambda_closure * closure_correctness
  + lambda_persistence * persistence_match
  + lambda_novelty * candidate_diversity
  - lambda_complexity * simplex_count_penalty
  - lambda_boundary * ||partial partial||_0
)
```

The useful test-time scaling experiment is to let the GFlowNet sample many
candidate future complexes and distill high-reward samples back into the
Soft-MoE experts.

## Split And Leakage Policy

For raw temporal hypergraphs:

- train on early timestamps;
- validate on later timestamps;
- test on the latest timestamps;
- hold out entire node families for inductive generalization where possible;
- ensure no future hyperedge leaks into closure negatives.

For transductive/inductive configs:

- preserve the source split labels;
- do not reassign rows with the general 80/10/10 splitter;
- report results separately from the main mixed corpus.

## Implementation Gates Before Approval

Before enabling these datasets:

1. add a license/terms note to `data/README.md`;
2. implement a `simplicial_hypergraph` task family;
3. add graph conversion tests for face/coface incidence;
4. add temporal leakage tests;
5. add small CPU-only persistence smoke tests;
6. add one bounded CUDA capacity test with simplex tokens enabled;
7. run an ablation against ordinary clique expansion without simplex tokens.

Only after these gates should the datasets be added to the default curation
manifest.

## Source Links

- OpenAI FrontierScience: <https://hf.co/datasets/openai/frontierscience>
- OpenAI HealthBench: <https://hf.co/datasets/openai/healthbench>
- OpenAI HealthBench Professional: <https://hf.co/datasets/openai/healthbench-professional>
- OpenAI GraphWalks: <https://hf.co/datasets/openai/graphwalks>
- tags-math-sx: <https://hf.co/datasets/SauravMaheshkar/tags-math-sx>
- NDC-substances-25: <https://hf.co/datasets/SauravMaheshkar/NDC-substances-25>
- tags-ask-ubuntu: <https://hf.co/datasets/SauravMaheshkar/tags-ask-ubuntu>
- Benson et al., Simplicial Closure: <https://arxiv.org/abs/1802.06916>
