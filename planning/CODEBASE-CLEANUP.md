# Codebase And Metrics Cleanup

## 2026-06-04 W&B Organization

The W&B surface had become difficult to use because several scripts logged the
same BPB/loss values under many historical aliases (`bpb/*`,
`openai_parameter_golf/*`, `competition/*`, `oai_competition/*`, `train/*`,
`val/*`) while advanced diagnostics logged hundreds of raw topology, toric,
complexity, and plot keys.

The cleanup rule is:

- keep raw historical keys in the run history for downstream scripts;
- mark raw historical namespaces hidden via `wandb.define_metric(...,
  hidden=True)`;
- add ordered visible aliases:
  - `00_primary/*` for the human scorecard;
  - `01_oai/*` for OAI/FineWeb BPB and test-time scaling;
  - `02_train/*`, `03_validation/*`, and `04_losses/*`;
  - `05_gflownet/*`, `06_graphcg/*`, `07_topology_geometry/*`,
    `08_toric_tropical_bgg/*`, and `09_complexity/*`;
  - `10_data_curriculum/*`, `11_artifact_size/*`, `12_optimization/*`,
    `13_system/*`, `14_phase_adaptive/*`, `15_analysis_media/*`, and
    `16_status/*`.

This preserves compatibility while making the default W&B panel order useful:
BPB, train/val loss, target gap, artifact size, full-dataset state, advanced
loss health, data mix, then plots and status.

## Checkpoint Cleanup Policy

Checkpoint cleanup is allowed, but the following are preserved:

- the active all-phases run directory;
- `best.pt`, `latest.pt`, `final.pt`, and export artifacts/manifests;
- OAI/FineWeb best-producing or low-BPB-capture lineages;
- the latest checkpoints in an active or recently useful run;
- representative milestone checkpoints at coarse intervals for ablation and
  recovery analysis.

The reusable pruning tool is `scripts/prune_parameter_golf_checkpoints.py`.
The first pruning target is the old dense native directory
`checkpoints/parameter_golf_oai_dense`, which contained 232 step checkpoints
and occupied about 58 GB. It is safe to prune step files there while retaining:

- `best.pt`;
- initial/final 16 MB artifacts and JSON manifests;
- latest 10 step checkpoints;
- explicit keep steps `1000`, `3000`, `4000`, and `14750`;
- every 5000-step milestone.

Applied cleanup:

```text
report: outputs/checkpoint_prune_parameter_golf_oai_dense_applied.json
deleted dense step checkpoints: 191
freed bytes: ~34.53 GB
kept steps: 1000, 3000, 4000, 5000, 10000, 14750, 15000, 20000,
  25000, 30000, 35000, 37750, 38000, 38250, 38500, 38750, 39000,
  39250, 39500, 39750, 40000
empty aborted checkpoint directories removed: 17
empty-directory report: outputs/empty_checkpoint_dirs_deleted_20260604T_wandb_cleanup.txt
```

Empty aborted checkpoint directories under `amelie-iska/parameter-golf/checkpoints`
may be removed after confirming they contain no files.

## Deferred Cleanup

Do not delete these without another inventory pass:

- current active checkpoints or analysis outputs;
- `outputs/post_resume_analysis/*` for recent runs until their synopsis and
  image/contact-sheet artifacts have been reviewed;
- `assets/2502.02617v1.pdf`, even though it is untracked, because it is still
  used for PolarQuant reference work;
- any `keys.txt` or secret material should remain local/untracked and must not
  be committed.
