# ToricBLM 25K Run Analysis and Next Full-Run Plan

Date: 2026-06-22

This note records the audit of the last long 25K-step ToricGT/OAI-baseline adaptation run and the planned changes for the ToricBLM scale-up path.  The analysis uses the exact run log and checkpoint artifacts, not campaign-level summaries.

## Run Artifacts

- Run directory: `runs/oai_sidecar/tg-bpb-bestof10-convextok2048-det-20260621T040717Z-bestfull-r011-gate2500_experimental_family_selective-20260621T174926Z`
- Exact log used: `runs/oai_sidecar/tg-bpb-bestof10-convextok2048-det-20260621T040717Z-bestfull-r011-gate2500_experimental_family_selective-20260621T174926Z/train.log`
- Final checkpoint: `checkpoints/tg-bpb-bestof10-convextok2048-det-20260621T040717Z-bestfull-r011-gate2500_experimental_family_selective-20260621T174926Z/tg-bpb-bestof10-convextok2048-det-20260621T040717Z-bestfull-r011-gate2500_experimental_family_selective-20260621T174926Z_step_025000.pt`
- Best prior short-run reference: the earlier 1K-step optimization loop produced a checkpoint around `0.49` BPB, while the 25K run finished near `0.84` validation BPB.

## Headline Measurements

The run produced 25,000 train rows, 12,500 metric-bearing rows, and one validation event at step 25,000.

Final values:

- Step 25,000 train loss: `1.1740`
- Step 25,000 train BPB: `0.8687`
- Step 25,000 validation loss: `1.4251`
- Step 25,000 validation BPB: `0.8374`
- Final int8+zlib roundtrip validation BPB: `0.83705590`
- Peak allocated VRAM: `12065 MiB`
- Total int8+zlib submission bytes: `17,274,122`, above the 16MB Parameter Golf cap.  This is acceptable for ToricBLM scale-up planning, but not for a competition submission artifact.

Train BPB windows:

| Window | Mean BPB | Min BPB | Max BPB | Last BPB |
|---|---:|---:|---:|---:|
| 1-1K | 0.7127 | 0.4928 | 5.0625 | 0.5241 |
| 1K-2.5K | 0.7775 | 0.4682 | 1.2444 | 0.7802 |
| 2.5K-5K | 0.7787 | 0.6357 | 0.9548 | 0.6905 |
| 5K-10K | 0.8151 | 0.5444 | 1.3912 | 0.9902 |
| 10K-15K | 0.9198 | 0.6640 | 1.6073 | 0.8329 |
| 15K-20K | 0.8892 | 0.7034 | 1.5341 | 0.9673 |
| 20K-25K | 0.8649 | 0.6780 | 1.1569 | 0.8687 |

The important behavior is the early valley: BPB reached short-run-quality values during the first 1-2.5K steps, then did not preserve them during the long run.

## Advanced-Metric Observations

The following correlations are against train BPB on metric-bearing rows.  They are time-confounded and should be treated as hypotheses rather than causal proof.

| Metric | First | Last | Mean | Corr. with train BPB | Interpretation |
|---|---:|---:|---:|---:|---|
| `graph_lm_bpb` | 6.2677 | 0.2587 | 0.2294 | +0.4740 | Graph stream became easy and saturated; high early values track warmup rather than late capability. |
| `oai_gfn` | 57.9754 | 0.2520 | 0.1796 | +0.5479 | GFlowNet loss improved strongly; reward stayed nonzero. |
| `gfn_R` | 0.0005 | 0.2093 | 0.2333 | -0.2460 | Better GFlowNet reward loosely aligns with lower BPB. |
| `oai_fot` | 36.5450 | 73.5928 | 32.9591 | +0.3103 | FoT objective became unstable/hard late. |
| `fot_R` | 0.0062 | 0.0001 | 0.1220 | -0.4422 | FoT reward collapsed almost to zero, an undesired mode collapse. |
| `fot_div` | 0.9766 | 0.0039 | 0.0141 | +0.2153 | Diversity collapsed; the positive correlation is confounded by high initial warmup diversity. |
| `mtp` | 7.6254 | 11.3232 | 10.4048 | -0.2424 | MTP remains high and noisy; its current scale is not cleanly explaining BPB. |
| `graphcg` | 0.0107 | 0.0141 | 0.0138 | +0.2983 | Full-rank GraphCG pressure may be too high or poorly staged for BPB in long runs. |
| `analogy` | 0.0661 | 0.0405 | 0.0520 | -0.2687 | Analogy training improved and appears compatible with lower BPB. |
| `tokengt_graph` | 0.9748 | 0.8854 | 0.9859 | -0.1080 | Graph tokenization loss improved modestly; no evidence it caused the BPB drift. |
| `memory` | 1.3831 | 0.8755 | 0.9092 | -0.0291 | Trajectory memory is noisy but not obviously harmful. |
| `toric` | 11.4453 | 3.2402 | 3.4280 | +0.0748 | Toric geometry diagnostics improved; weak BPB relation. |
| `vector_bundle_1d_cone` | 0.7810 | 0.6129 | 0.6041 | -0.0226 | Stable, weakly coupled. |
| `bgg` | 0.1373 | 0.0026 | 0.0357 | -0.3895 | BGG certificate loss became too easy/saturated.  Useful as a metric, not a late heavy objective. |
| `koszul` | 0.0348 | 0.0406 | 0.0435 | +0.2263 | Koszul residual did not improve; keep small and curriculum-gated. |
| `cca` | 0.8246 | 1.4757 | 1.4508 | -0.0278 | Combinatorial commutative algebra loss appears mismatched or too hard at current staging. |
| `derived` | 3.7961 | 0.0003 | 0.1797 | -0.2181 | Derived signature became saturated like BGG. |

## Desired Behaviors

1. Early BPB drops rapidly with ConvexTok-2048, graphification, and first-class TokenGT structure active.  This confirms the adapted OAI baseline can exploit the tokenizer and graph features.
2. GFlowNet reward remains nonzero after warmup and has a weakly favorable relation with BPB.  This supports keeping embedding-space stochastic search pressure.
3. Analogy loss improves steadily and does not show an obvious BPB penalty.  This supports preserving analogical memory retrieval as a trainable component.
4. Toric geometry and vector-bundle/one-dimensional-cone metrics remain numerically stable.  They can stay as low-weight regularizers and audits.
5. BGG and derived signatures become highly predictable.  That is useful for interpretability, but the loss should not dominate after saturation.

## Undesired Behaviors

1. The model does not preserve the best early BPB valley during long training.  Validation was too sparse to select the best intermediate checkpoint.
2. FoT reward and diversity collapse.  `fot_R` falls to about `0.0001` and `fot_div` to `0.0039` by the end, while `oai_fot` rises above `70`.
3. Graph LM BPB becomes nearly trivial on the curated graph stream.  It is no longer providing enough hard reasoning signal late in training.
4. Combinatorial commutative algebra and Koszul objectives do not visibly improve late.  Their weights should remain small and be activated only when their exact audits are informative.
5. The final int8+zlib artifact is above the competition cap.  That matters for Parameter Golf submission, but the ToricBLM scaled model is intentionally outside the 16MB regime.

## Causal Hypotheses

1. **FoT collapse is the dominant long-run reasoning failure.**  Its reward goes to zero, diversity collapses, and its loss grows.  The current reward target and branching/depth setting likely stop providing valid high-reward branches once the base language model distribution changes.
2. **The graph auxiliary stream is too easy before the end.**  Graph LM BPB near zero means the graph stream no longer forces new structure.  Adding late train-only codex5.5 Tree/Forest-of-Thought records should restore hard graph reasoning pressure.
3. **Auxiliary objectives are not stage-specific enough.**  BGG and derived losses become saturated, while FoT and CCA remain difficult.  These should be split into audit, warmup, and late-hard curricula rather than kept as one flat bundle.
4. **The long-run optimizer schedule is tuned for short-run BPB discovery.**  The best BPB occurs early.  A larger model should use mu-transfer with lower effective matrix LR, frequent validation, and checkpoint selection by validation BPB rather than only final-step metrics.
5. **Scale should mainly be width, not depth.**  muP is strongest for width transfer.  Depth transfer is possible but less reliable; the first scaled ToricBLM run should preserve the existing training dynamics as much as possible while increasing width.

## Planned Updates for the Next Full Run

1. Add `AmelieSchreiber/codex5.5_ToT` as a train-only late-stage graph stream.  It will be represented as graph records and mixed into both graph LM and sidecar batches after the late-stage step threshold.
2. Upsample codex5.5_ToT to three passes by materializing a train-only 3-pass Parquet view.  The validation/test BPB stream remains unchanged.
3. Keep FoT active, but lower the long-run FoT weight and monitor reward/diversity directly.  Use late-stage codex5.5 ToT/FoT records to prevent reward collapse rather than simply increasing the FoT loss weight.
4. Keep GFlowNet active with modest weight because it remains compatible with lower BPB.
5. Reduce late heavy pressure from saturated BGG and derived signatures.  Keep them as metrics and small regularizers.
6. Keep GraphCG full-rank, but reduce the default pressure relative to BPB and use BPB-orthogonal routing rather than raw full-rank pressure.
7. Increase validation/checkpoint cadence to every 1K steps for long runs.  Preserve the early valley if it reappears.
8. Use muP/mu-transfer for the scaled model so optimizer settings are inherited from the best small-run family without arbitrary LR guessing.
9. Treat the 16MB competition constraint separately from ToricBLM scale-up.  The scaled run optimizes model quality and biomedical reasoning preparation, not Parameter Golf submission bytes.

## Ready-State Criteria Before Restart

- [x] The codex5.5_ToT train-only 3-pass graph stream exists and can be read by `GraphParquetTokenStream`.
- [x] The OAI baseline trainer can switch to `ScheduledGraphParquetTokenStream` at the configured late step.
- [x] A muP base-shapes file exists for base/delta/target models.
- [x] A full ToricBLM scale-up env config exists and records the exact data paths, model shape, LR scaling, and late-stage schedule.
- [x] A smoke test compiles the changed trainer and verifies a scheduled graph batch.
- [x] A smoke test verifies trainer-side muP base-shape application on the 160M target model.

The scaled full run should be started only after these criteria are satisfied.
