# Cleanup Ledger - 2026-06-19

Generated: `2026-06-19T23:36:10Z`.

## Active Training Chain

- tmux session: `toricgt_convextok_full_train`
- active step before cleanup: ConvexTok-2048 full dataset export, followed by the 1K-step BPB campaign loop.
- active external output is intentionally not deleted: `/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/parameter_golf_convextok2048_det_full`.

## Cleanup Policy

- Keep source code, docs, papers, planning files, train logs, markdown reports, campaign state, and compact selected int8 artifacts.
- Keep no more than five high-priority checkpoint binaries.
- Delete redundant full-precision checkpoints, redundant full `final_model.pt` outputs, noisy stderr logs, generated full-analysis HTML payload directories under `training_notes`, and Python caches.
- Preserve hyperparameter/outcome evidence below from `runs/oai_sidecar/*/train.log`.

## Best Historical BPB Runs From Local Logs

| rank | run | step | val BPB | int8 roundtrip BPB | int8 submission bytes |
|---:|---|---:|---:|---:|---:|
| 1 | `toricgt-oai-sidecar-bpb112-campaign-20260616T222606Z-run001-large_batch_lower_lr_sidecar_light-20260616T222607Z` | 5000 | 1.2618 | 1.2637899 | 14959389 |
| 2 | `tg-bpb119-fixed-cas-tmux-20260618T133107Z-r004-gate1500_fast_main_lr_aux_conflict_recovery-20260618T185332Z` | 1500 | 1.2969 | 1.29915775 | 16041801 |
| 3 | `tg-bpb119-radius4-recovery-20260618T025622Z-r005-gate1500_fast_main_lr_aux_conflict_recovery-20260618T101823Z` | 1500 | 1.2978 | 1.30025528 | 16063729 |
| 4 | `tg-bpb119-fixed-cas-tmux-20260618T133107Z-r002-gate1500_fast_main_lr_aux_conflict_recovery-20260618T151733Z` | 1500 | 1.2983 | 1.30057225 | 16070902 |
| 5 | `tg-bpb119-fixed-cas-tmux-20260618T133107Z-r003-artifact_margin_conservative_aux-20260618T170759Z` | 1500 | 1.3038 | 1.3056297 | 15153671 |
| 6 | `tg-bpb119-radius4-recovery-20260618T025622Z-r003-gate1500_fast_main_lr_aux_conflict_recovery-20260618T063726Z` | 1500 | 1.3042 | 1.30639114 | 15898944 |
| 7 | `tg-bpb119-fixed-cas-tmux-20260618T133107Z-r001-artifact_margin_conservative_aux-20260618T133107Z` | 1500 | 1.3049 | 1.30677229 | 15145778 |
| 8 | `tg-bpb119-fixed-cas-tmux-20260618T133107Z-r005-artifact_margin_conservative_aux-20260618T204333Z` | 1500 | 1.3067 | 1.30839432 | 15166462 |
| 9 | `tg-bpb119-radius4-recovery-20260618T025622Z-r001-gate1500_fast_main_lr_aux_conflict_recovery-20260618T025624Z` | 1500 | 1.3094 | 1.31169923 | 15904555 |
| 10 | `tg-bpb119-oai-transfer-gfn-mtp-20260617T235313Z-r001-gate1500_fast_main_lr_light_graphcg-20260617T235417Z` | 1500 | 1.3107 | 1.31278244 | 15602833 |
| 11 | `tg-bpb119-radius4-recovery-20260618T025622Z-r004-artifact_margin_conservative_aux-20260618T083017Z` | 1500 | 1.3115 | 1.31320402 | 15095315 |
| 12 | `tg-bpb119-radius4-recovery-20260618T025622Z-r002-artifact_margin_conservative_aux-20260618T044911Z` | 1500 | 1.3149 | 1.31691433 | 15078423 |
| 13 | `tg-bpb119-gate1500-routed-curriculum-20260617T213617Z-r001-gate1500_high_batch_toric_bgg_memory-20260617T213618Z` | 1500 | 1.3175 | 1.31960777 | 15303557 |
| 14 | `tg-bpb119-1k-ideas14-20260619T152338Z-r004-gate1500_fast_main_lr_aux_conflict_recovery-20260619T190741Z` | 1000 | 1.3288 | 1.33051878 | 15122188 |
| 15 | `tg-bpb119-1k-ideas14-20260619T152338Z-r002-gate1500_fast_main_lr_aux_conflict_recovery-20260619T163816Z` | 1000 | 1.3294 | 1.33065348 | 15175534 |
| 16 | `tg-fot-bpb119-1k-20260618T223154Z-r008-gate1500_fast_main_lr_aux_conflict_recovery-20260619T072216Z` | 1000 | 1.3298 | 1.33158715 | 15134519 |
| 17 | `tg-fot-bpb119-1k-20260618T223154Z-r010-gate1500_fast_main_lr_aux_conflict_recovery-20260619T095319Z` | 1000 | 1.3325 | 1.33425012 | 15103615 |
| 18 | `tg-fot-bpb119-1k-20260618T223154Z-r012-gate1500_fast_main_lr_aux_conflict_recovery-20260619T122312Z` | 1000 | 1.3328 | 1.33469029 | 15129387 |
| 19 | `tg-fot-bpb119-1k-20260618T223154Z-r004-gate1500_fast_main_lr_aux_conflict_recovery-20260619T022035Z` | 1000 | 1.3334 | 1.33516988 | 15124624 |
| 20 | `tg-fot-bpb119-1k-20260618T223154Z-r002-gate1500_fast_main_lr_aux_conflict_recovery-20260618T234752Z` | 1000 | 1.3336 | 1.33513082 | 15161968 |

## Kept Checkpoints

- `checkpoints/parameter_golf_oai_dense/best.pt`
- `checkpoints/tg-bpb119-1k-ideas14-20260619T152338Z-r004-gate1500_fast_main_lr_aux_conflict_recovery-20260619T190741Z/tg-bpb119-1k-ideas14-20260619T152338Z-r004-gate1500_fast_main_lr_aux_conflict_recovery-20260619T190741Z_step_001000.pt`
- `checkpoints/tg-bpb119-1k-ideas14-20260619T152338Z-r005-gate1500_fast_main_lr_light_graphcg-20260619T202250Z/tg-bpb119-1k-ideas14-20260619T152338Z-r005-gate1500_fast_main_lr_light_graphcg-20260619T202250Z_step_001000.pt`
- `checkpoints/tg-bpb119-radius4-recovery-20260618T025622Z-r005-gate1500_fast_main_lr_aux_conflict_recovery-20260618T101823Z/tg-bpb119-radius4-recovery-20260618T025622Z-r005-gate1500_fast_main_lr_aux_conflict_recovery-20260618T101823Z_step_001500.pt`
- `checkpoints/toricgt-oai-sidecar-bpb112-campaign-20260616T222606Z-run001-large_batch_lower_lr_sidecar_light-20260616T222607Z/toricgt-oai-sidecar-bpb112-campaign-20260616T222606Z-run001-large_batch_lower_lr_sidecar_light-20260616T222607Z_step_005000.pt`

## Deleted Counts

- checkpoint `.pt` files queued for deletion: 83
- full precision run `final_model.pt` files queued for deletion: 32
- redundant int8 run artifacts queued for deletion: 28
- generated full-analysis directories queued for deletion: 38
- generated Codex stderr logs queued for deletion: 29
- Python cache directories queued for deletion: 8

## Notes

The deleted artifacts are reproducibility byproducts, not implementation sources. The reports and train logs remain as the canonical record of hyperparameters, outcomes, and controller decisions.
