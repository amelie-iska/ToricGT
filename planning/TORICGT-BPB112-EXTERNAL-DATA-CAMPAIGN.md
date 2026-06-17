# ToricGT BPB <= 1.12 External-Data Campaign

This campaign is ToricGT-only.  Code, configs, checkpoints, analyses, W&B runs,
and Codex review hooks live under `/home/iska/Documents/amelie/bio/ToricGT`.
The path `/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data` is used
only as an external dataset source for ToricGT parquet shards.

## Objective

Reach OpenAI Parameter-Golf-style byte BPB `<= 1.12` while preserving the
ToricGT reasoning stack:

- sequential byte prediction as the primary BPB objective;
- TokenGT-style prefix-causal graph tokenization and graph-logit fusion;
- full-rank GraphCG from step 0;
- embedding-space graph-of-thought GFlowNet policy training;
- toric/tropical geometry probes and CAS certificates;
- vector-bundle and sheaf probes;
- GUDHI persistent-homology audits, vectorized PH features, and Macaulay2
  bigraded persistence-module checks;
- Toric BGG / category O diagnostics and low-weight training pressure;
- trajectory memory and analogical retrieval diagnostics;
- exact toric embedding sidecars and branching graph-of-thought HTML reports.

## Active Config

Use:

```bash
config/train.parameter_golf_all_phases_external_toricgt_bpb112.yaml
```

Important configuration choices:

- External ToricGT dataset:
  `/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/curated_hf_shards/{train,validation,test}/*.parquet`.
- ToricGT-local FineWeb/OAI token data:
  `amelie-iska/parameter-golf/data/datasets/fineweb10B_sp1024/*.bin`.
- `analysis.periodic_interval_steps: 5000`, matching the intended automated
  check-in cadence of roughly five hours if the fast profile reaches about
  1000 optimizer steps per hour.
- `analysis.geometry_branches: 12` and
  `analysis.branching_reasoning_max_nodes: 240`.
- `analysis.gudhi_persistence` uses 4 records, 24 points, 7 radius levels, and
  7 reasoning levels per periodic audit.
- Advanced losses are instantiated and nonzero from step 0, but kept small
  enough that BPB remains the primary optimization signal.
- `fineweb_mix_ratio: 0.94` in the first 5K-step warmup keeps the competition
  distribution dominant while admitting curated graph/reasoning data from the
  start.
- Microbatch `2` at 2048 context filled the 24 GB card on the first forward
  pass, and microbatch `1` with accumulation `48` was stable but took about
  one minute per optimizer step.  The active speed profile is now
  `max_seq_len: 1024`, `batch_size: 1`, and `grad_accum_steps: 8`, so each
  optimizer step scores 8192 bytes before auxiliary graph/reasoning work.
  The goal is about 1000 optimizer steps per hour and periodic Codex/analysis
  review every 5000 steps.

## Automation Contract

The active ToricGT supervisor for the OAI baseline sidecar campaign is:

```bash
scripts/run_oai_sidecar_bpb_campaign.py
```

It must:

1. launch one fresh OAI-baseline ToricGT-sidecar run from step 0;
2. train exactly 5000 steps with `TRAIN_SEQ_LEN=1024`,
   `TRAIN_BATCH_TOKENS=524288`, and all ToricGT sidecar metrics active;
3. save a step-5000 checkpoint and int8+zlib artifact round-trip BPB;
4. run `scripts/run_oai_sidecar_full_iteration_analysis.py` before launching
   any next run;
5. export every W&B metric available for the run via
   `scripts/analyze_wandb_metrics.py`;
6. extract real OAI checkpoint hidden states via
   `scripts/extract_oai_sidecar_embeddings.py`;
7. run exact GUDHI/Macaulay2 persistence audits, vectorized PH dashboards,
   Sage/Macaulay2 toric embedding sidecars, toric embedding reports,
   vector-bundle/sheaf reports, Toric BGG/category O reports, branching
   reasoning simplex-tree reports, static screenshots, and browser screenshots;
8. validate exactness with `scripts/validate_analysis_exactness.py`;
9. write `FULL-ITERATION-REPORT.md`, `next_profile_decision.json`, an HTML
   index, and command logs under
   `training_notes/<campaign>/RUN-*/`;
10. use the analyzer's `next_profile_hint` as the next fresh-start profile;
11. stop instead of relaunching if strict analysis or training fails.

Codex reviews should inspect the generated HTML/screenshots, W&B metrics, OAI
BPB, exact PH/CAS reports, sidecar losses, and the next-profile proposal.  The
next run must not be launched until the complete report and exactness checks are
available.

## Review Cadence

- Periodic analysis: after every fresh 5000-step attempt.
- BPB gate: step 5000.
- Primary target: OAI competition BPB `<= 1.12`.
- Review cap: 25 fresh runs for the first campaign; if the target is missed,
  write a synopsis and start a second 10-run campaign only after incorporating
  the findings.

If 25 full restart attempts fail to reach `<= 1.12`, write an aggregate
synopsis before trying another 10-run campaign.  The synopsis should identify
the best BPB windows, the auxiliary-loss families that correlated with BPB
improvement/regression, and candidate adaptive schedules for the next campaign.

## Safety

- Do not move ToricGT code or training orchestration into TropicalGT.
- Do not delete user data or the external dataset.
- Do not train on validation/test shards except through score-first evaluation
  or declared analysis sidecars.
- Keep the Parameter-Golf artifact cap visible; export failures are hard
  failures even if raw checkpoints improve BPB.
- Do not leave training inactive after a periodic review; the supervisor or
  fallback continuation must keep the run advancing unless the BPB target or
  iteration cap is reached.
