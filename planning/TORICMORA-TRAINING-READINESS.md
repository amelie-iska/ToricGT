# ToricMora training-readiness audit

November 18, 2026. Reviewed the Iska `ToricMora` checkout at base commit `004e04057cf834028c495c0ca2a319d7ef0e7ed8` and the open acceptance items in [the integration plan](TORICMORA.md). This pass used source review and authored symbolic and general-text fixtures. It opened no actual corpus or existing model weights, performed no optimizer updates, and launched no training. The conditioned biomolecular generation and biomedical knowledge-graph items in the supplied audit are outside this implementation.

## Implemented and verified

| Item | Implementation | Evidence and scope |
| --- | --- | --- |
| Transitive split isolation | [Symbolic manifest audit](../src/toricgt/toricmora_symbolic_splits.py) groups source IDs, compatible canonical question and answer pairs, and reviewed similarity links. Fixed holdouts propagate through the component. Declared validation/test conflicts quarantine the component. | Authored controls preserve MATH-500's test role and one group per OpenR1 problem. Actual training membership and retrieval index exposure have separate fields and checks. An omitted inventory is reported as unverified. The mixed-source ingestion path was outside this change. |
| Transported graph identity | [Native tokenizer](../src/toricgt/graph_tokenizer.py) accepts stable node and edge IDs as optional inputs. It carries node identities into toric coordinates and directed endpoint embeddings and retains explicit padded edge slots. The [native model](../src/toricgt/model.py) uses transported IDs for LM position fallback and reveal-rank tie ordering. | Independently permuted node and edge rows give aligned logits and embeddings with softmax and hybrid attention. Future-rank perturbation leaves earlier logits bit-identical in both checks. This verifies the GraphBatch-to-output boundary with transported features. The legacy graph JSON path still constructs its own row-indexed features. |
| Crash-safe checkpoint writes and native generated-graph recovery | The [checkpoint helper](../src/toricgt/synthetic_resume.py) writes and verifies a sibling archive before atomic replacement. The [trainer](../scripts/train.py) has an opt-in CUDA lane that binds the absolute learning-rate horizon and configuration digest and restores model, optimizer, AMP scaler, and Python, NumPy, Torch CPU and selected CUDA RNG state. This lane uses generated batches, so its next draw is in the RNG state. | An authored CPU no-update simulation reproduces the next generated batch, dropout forward result, stored momentum, scaler and LR. Simulated partial writes and invalid archives preserve the previous file. The CLI rejects data-backed inputs before opening them. The full CUDA optimizer-interruption and stream-backed cursor tests remain training gates. |
| Scoped analogies and topology | [Forest consensus](../src/toricgt/toricmora_analogy.py) counts one matching root signature per source forest with deterministic tie handling. The branching, GUDHI, flag-chain and standalone geometry auditors require every deduplicated simplex image to be a target face. | Duplicate roots cannot outvote a true majority. A triangle collapsing to two nonadjacent target vertices fails exact certification. A collapse onto a real target edge passes. GUDHI checks vertices and counts valid edge collapses. Historical reports predate this rule. |
| Bounded visible-prefix search | [Symbolic search control](../src/toricgt/toricmora_symbolic_search.py) compares single path, best-of-three, ToT, FoT and FoT with analogy using the same arithmetic forest prompts, action proposer, terminal verifier and per-case caps. | Six frozen queries include one infeasible case, one ambiguous case and an unseen balanced shape. Caps are 24 operator tokens, 12 parent expansions and 8 verifier calls per case. Actual work differs by policy and is recorded. The deterministic proposer establishes the control and trace format for later model comparisons. |
| Discrete trajectory-balance semantics | The [symbolic trajectory verifier](../src/toricgt/symbolic_trajectory_balance.py) checks normalized forward and successor-conditioned backward transitions, explicit completion and fixed positive rewards on an enumerable DAG. | At Z = 7, all three complete paths satisfy trajectory balance. Terminal masses are exactly 2/7 and 5/7 for rewards 2 and 5. An incorrect reverse policy can preserve terminal totals while failing individual paths, and the verifier rejects it. The existing native pooled auxiliary is still a diagnostic and is outside this exact DAG implementation. |
| Byte accounting | [Graph-node BPB evaluation](../scripts/evaluate_tokengt_fineweb_bpb.py) and [native logging](../scripts/train.py) separate fallback estimates from complete tokenizer-backed causal BPB. Mixed or missing byte coverage suppresses byte-normalized metrics while retaining bits per token. | A manually constructed SentencePiece fixture accounts for 82 bytes across 30 scored targets. A missing tokenizer produces a proxy with the exact BPB field unavailable. A partial-coverage check suppresses the invalid aggregate. |

## Search comparison

Each entry reports cases with at least one verified solution out of six, including the infeasible case. The final two rows show case-specific distinct solutions across the suite. The runs start from authored arithmetic prompts whose answer sets are checked by exhaustive enumeration in the focused test.

| Seed and measure | Single path | Best-of-three | ToT | FoT | FoT with analogy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 verified cases | 1 | 2 | 3 | 3 | 4 |
| 29 verified cases | 3 | 3 | 5 | 4 | 4 |
| 17 distinct verified solutions | 1 | 3 | 4 | 4 | 5 |
| 29 distinct verified solutions | 3 | 3 | 6 | 5 | 5 |

The analogy variant misses the unseen balanced root at both seeds. Total generated operators at each seed are 12, 36, 78, 114 and 114, respectively. These are common budget caps. Learned transfer and equal-consumption comparisons require separate evidence.

## BPB interpretation

For scored targets T, let l_i = -log p(y_i | visible prefix_i) and b_i be the number of encoded bytes attributed to that target. The reported aggregate is

```text
bits_per_token = sum(i in T, l_i) / (ln(2) * |T|)
bits_per_byte  = sum(i in T, l_i) / (ln(2) * sum(i in T, b_i))
```

The numerator and denominator cover the same targets. A fallback denominator produced from tokens such as `<sp17>` can support a diagnostic proxy only. For comparison with an official baseline, additionally pin the tokenizer, evaluation split, scoreable target bytes and visible-prefix factorization. A graph serialization must specify what is coded, the node and edge order, and whether graph structure is supplied as context or encoded as part of the payload. Node-attribute MSE, bits per graph event, and text bits per byte measure different quantities. This pass verified accounting and did not measure improvements.

## Recorded campaign review

The checked-in [24+2 roster](campaign26_status.json) records 24 distinct proxy cells covering the complete Cartesian grid and both required target seeds. All recorded scores are finite. Recomputing the pair averages confirms the stored winner and both targets carry the winner's settings. No campaign artifacts were used to verify the roster.

The checked-in [campaign runner](../scripts/run_mup_campaign26.py) implements a synthetic token and geometry experiment. Its validation loss combines random-token cross entropy and a structure auxiliary. Exponentiating that combined loss gives a campaign-specific score. It does not establish natural-language perplexity, held-out BPB, arithmetic accuracy, optimizer transfer under MuAdamW, or corpus generalization. Those claims need independent measurements using the relevant recipes and data. The planning note `No campaign run has been started` precedes this roster.

## Verification and deferred gates

The existing Iska `tokengt` environment passed 103 focused tests with CUDA and external reporting disabled. The test set covered native graph identity and causal routing, model shapes, checkpoint recovery and failure injection, split controls, typed analogies, symbolic search and trajectory balance, simplex checks, and graph BPB. The authored SentencePiece fixture emitted two SWIG deprecation warnings. `git diff --check` and the trainer's CLI admission check passed.

## Prepared-stream follow-up

The native trainer now has an exact-recovery path for reviewed symbolic graph Parquet. Admission binds an independently supplied manifest SHA-256 to the review metadata, frozen Parquet ordering, and a separate retrieval inventory. It checks the real Parquet record IDs against the symbolic split audit, checks validation membership independently, and rejects unknown, duplicate, held-out or retrieval-ineligible IDs in the relevant streams. The review supplies source identities and similarity references. The operator supplies the pin and the retrieval IDs exported from the index they intend to use. An empty explicit retrieval list represents a run with no retrieval exposure.

Use one Parquet file per split with string `record_id` and `graph_json` columns, in that order. Each graph carries the same record ID as its row. The envelope uses schema `toricgt-prepared-symbolic-v1` and keys `review`, `train`, `validation` and `retrieval`. Each non-null entry has a relative `file` and its lowercase `sha256`. The optional validation entry is explicitly null when absent. The review JSON has `records` and `similarity_links` arrays using the existing symbolic audit schema. The independent retrieval JSON has one `record_ids` array.

The admission-only command checks bytes, schema and membership before returning aggregate counts. It exits before opening a checkpoint or allocating a model. Individual graph JSON records are decoded as the stream consumes their batches. Startup reads the sealed file bytes to verify hashes and scans just the ID column for inventory checks. Recovery then seeks directly to the row group that holds the next batch, including a short last batch or the first batch of the next epoch. It does not replay the earlier graph payloads.

```sh
PYTHONPATH=src python scripts/train.py --check-prepared-stream \
  --prepared-stream-manifest "$MANIFEST" --prepared-stream-sha256 "$MANIFEST_SHA256"
```

An authorized CUDA run uses the same manifest and digest without the admission-only flag. A segmented run supplies `--schedule-total-steps` to fix the LR horizon in advance. On resume, `--steps` is the remaining segment length and the saved horizon is inherited unless the exact same value is supplied. The next batch count must agree with the saved optimizer-step boundary and accumulation count. The atomic checkpoint binds the manifest content, resolved recipe, CUDA runtime settings, optimizer buffers, AMP scaler and Python, NumPy, Torch CPU and selected CUDA RNG states. The prepared lane uses fixed Parquet ordering and offline logging. The ordinary curated streams and generated-graph recovery retain their own entry points.

## CUDA interruption parity

A no-update test used the existing Iska RTX 4090 and PyTorch 2.8.0 with CUDA 12.8. Seed 29 initialized a tiny native hybrid-attention model, authored FP16 scaler and AdamW buffers, and five reviewed authored training rows across three Parquet row groups. A second fresh process initialized with a different seed, restored the checkpoint and resumed after two consumed microbatches. Both processes then observed the short last batch and the first batch after wraparound. They matched bit for bit on pre-update weights, the next CPU and CUDA random draws, inputs, dropout-active logits, accumulated scaled gradients, optimizer buffers, scaler, LR and cursor. The fixture changes no model weights and calls no optimizer step.

The deterministic CUDA profile enables deterministic operations, fixes the cuBLAS workspace and selects full-precision FP32 matmul settings. The result applies to this fixture and runtime. The end-to-end optimizer-update parity gate for a real interrupted run remains open. [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.15/notes/randomness.html)

## Checkpoint-backed symbolic evaluation

The arithmetic forest suite now accepts a trusted local native causal LM checkpoint and its tokenizer with separately pinned hashes. The checkpoint scorer receives the visible puzzle, root and operator prefix. Each legal operator must encode as one additional distinct token after the full prompt. Incompatible tokenizers and oversized prompts fail before scoring. The report keeps the same caps on generated operators, parent expansions and verifier calls across policies and records actual model calls and input tokens. The model input token cap is the per-prompt limit times the maximum expansion count. The existing seeded comparator remains available.

The authored checkpoint smoke evaluated all five methods across six frozen queries from random initialization. Its one-expansion cap produced one model scoring call per case and method. The check proves the evaluation route and budget accounting. It does not establish learned reasoning quality.

## Follow-up verification and open measurements

The combined focused suite used only authored fixtures. It passed 112 CPU tests with the CUDA test opted out. The separate CUDA parity test passed with the existing GPU lease. No actual corpus or existing checkpoint was opened, no optimizer updates ran, and no training started.

Before claiming production readiness, run admission on authorized reviewed artifacts, verify the retrieval inventory against its actual index export, compare optimizer-update continuation on the intended CUDA recipe, and evaluate a reviewed trained checkpoint at the frozen search and BPB budgets. The prepared stream now supports seeking and admission for its native symbolic graph format. The checks here establish fixture behavior, input contracts and pre-update parity.


## GPU training goal and iteration plan

The active goal covers native ToricMora training on an authorized symbolic prepared stream, followed by checkpoint and evaluation verification.

- [x] Verify admission, exact prepared-stream recovery, and CUDA interruption through the next gradients on authored fixtures.
- [x] Verify the native CUDA AdamW update boundary in separate processes using fixed authored gradients and disposable parameter tensors. Both processes produce bit-identical parameters, optimizer moments, LR and AMP scaler state over a clipped finite update, an overflow that skips the update, and a finite update after overflow. The first parameter's AdamW counter is 3, 3 and 4 after the three attempts. AMP scale is 32, 16 and 32. These fixture updates consume no training examples or real model weights.
- [ ] Admit the operator-reviewed symbolic manifest and its SHA-256 on Iska. The artifact search found no applicable approved manifest for this native trainer. Admission therefore has not been run against production artifacts.
- [ ] Freeze a GPU recipe, seed, epoch boundaries, validation cadence and fresh output directory using the admitted inputs. Use the existing Iska environment and GPU lease. Check that the resulting schedule consumes the intended batches over two epochs and that checkpoints align with actual accumulation boundaries.
- [ ] Start the first scheduled segment, verify progress and save an atomic boundary checkpoint. Resume under the same pinned plan and verify the final cursor, completed steps, finite metrics and checkpoint integrity. If a run is interrupted, retain its receipts and resume from the latest verified boundary.
- [ ] Evaluate the final checkpoint on the eligible held-out inputs and record what the native graph objective supports. Use the pinned checkpoint-backed search gate for a compatible causal language checkpoint. Report graph validation and language evaluation separately.

The two focused CUDA interruption tests and 13 CPU checks passed. No corpus training has started. The optimizer test closes arithmetic parity on authored tensors and leaves real-runtime model quality and checkpoint-backed held-out results open.


### Two-epoch launch plan

The [prepared-run planner](../scripts/plan_prepared_training.py) verifies admission and emits the two CUDA invocations without creating any output files. It selects the largest accumulation divisor within the requested cap that places each epoch boundary on an optimizer-step boundary, counts a short final batch, and pins the same schedule and seed across the two segments. The second segment resumes from the first segment's final checkpoint and uses a separate output directory. The receipt lists the expected checkpoint step and stream cursor at each boundary. When a validation split exists, the plan evaluates all of its batches at both boundaries.

```sh
PYTHONPATH=src python scripts/plan_prepared_training.py \
  --manifest "$MANIFEST" --sha256 "$MANIFEST_SHA256" --output "$NEW_RUN_DIR"
```

The authored plan test covers five training rows, two rows per batch and an accumulation cap of two. It selects one microbatch per update, three updates per epoch, and a six-update schedule. It also verifies that the planner refuses an existing output directory and writes no files. The focused CPU suite passed 14 tests. A real receipt still requires the approved manifest and pin.

The default CUDA recipe also passed an FP16 forward and gradient capacity check at its padded graph limits. A batch of four graphs with 256 nodes and 1,024 edges each used a 6,084,723-parameter model and peaked at 15,496 MiB allocated and 15,592 MiB reserved. Gradients were finite and parameter version counters stayed unchanged. Seed 29 and the existing GPU lease were used. This measures a single authored microbatch before optimizer state allocation and production data access.


### Completion checks

Prepared checkpoints now record their manifest pin and the latest validation step, record count and masked MSE. The [run verifier](../scripts/verify_prepared_training.py) reads the two local checkpoints named in the trusted launch plan. It checks archive integrity, the expected epoch and stream cursor boundaries, the shared recipe and input pins, optimizer state, finite weights and completion of the planned validation split. It computes each checkpoint SHA-256 without opening the corpus.

```sh
PYTHONPATH=src python scripts/verify_prepared_training.py --trusted-plan "$PLAN_JSON"
```

The focused CPU suite passed 22 tests. Authored checkpoint controls rejected a changed cursor, recipe drift, an incomplete validation record count, and unchanged epoch-two weights. The CUDA optimizer checks above were separately verified. Production admission and both training epochs remain pending the operator-reviewed symbolic manifest and its SHA-256.

## November 19 review boundary

The current [review handoff](TORICMORA-REVIEW-HANDOFF.md) supersedes the launch target above until the owner approves the filtered input. The code and a public symbolic candidate are staged for review.
