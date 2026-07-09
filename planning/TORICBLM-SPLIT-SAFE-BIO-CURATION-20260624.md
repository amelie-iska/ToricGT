# ToricBLM Split-Safe Bio Curation Gate

Status: v2 parallel curation active as of 2026-06-24.

## Decision

Training is intentionally stopped until the biological graph/FoT streams satisfy
the leakage and diversity constraints below.  The previous full raw run mixed
ProTrek-split protein shards with raw upstream `raw_hf_bio_scale/*/train`
directories.  That was not split-safe for DNA, RNA/Rfam/RNAcentral, PubChem, or
raw UniRef50 rows, so those raw globs have been removed from the training config.

## Required Training Inputs

- Protein rows: use ProTrek-derived split shards where available.
- Protein coordinate rows: use AFDB/PDB/PubChem3D coordinate-bearing shards.
- Non-protein DNA/RNA/small molecules: use only the newly curated
  `data/toricblm_nonprotein_fot_splits/v2_parallel/*/train/*.parquet`
  shards.
- Raw upstream files under
  `/home/iska/Documents/amelie/bio/iska-net/data/raw_hf_bio_scale` are curation
  inputs only, not direct training inputs.

## Implemented Guards

- `TORICBLM_REQUIRE_SPLIT_SAFE_BIO=1` in
  `configs/toricblm_mup_170m_convextok8192_biomed_fot_structure.env`.
- `scripts/launch_toricblm_mup_full_codex55.sh` now refuses to launch if
  `GRAPH_TRAIN_GLOB` or `LONG_ENTRY_TRAIN_GLOB` contains `raw_hf_bio_scale`.
- The same launch preflight requires
  `data/toricblm_nonprotein_fot_splits/v2_parallel/manifests/toricblm_nonprotein_similarity_fot_split_manifest.json`
  and at least `450000` train rows each for `dna`, `rna`, and
  `small_molecule`.
- `TORICBLM_REQUIRE_FOT_FORMAT=1` now also requires sampled graph-training
  Parquet shards to contain non-empty `graph_json`, `forest_json`,
  `thought_forest_json`, `convextok_dag_json`, and `training_views_json`.
  The preflight checks the canonical thought-forest schema
  `toricgt.biomed_source_grounded_forest_of_thought.v1` and the ConvexTok DAG
  schema `toricgt.convextok_tokenization_dag.v1`.

## Active Curation Jobs

The old sequential non-protein curation path was stopped.  It was correct but
too slow because it scanned highly clustered source shards after hitting small
per-cluster caps.  The replacement path is plan-parallel:

- output directory:
  `data/toricblm_nonprotein_fot_splits/v2_parallel`;
- separate tmux workers for DNA, RNA, and small molecules;
- bounded graph/FoT/ConvexTok row materialization workers inside each plan;
- sequence MinHash signatures computed from bounded sketches rather than every
  k-mer in very long rows;
- per-cluster cap increased to `256` for this downsampled pass, which preserves
  diversity but prevents pathological skipping through sorted source shards;
- no train-only early stop on the full v2 pass, so validation and test splits
  are also filled by the same cluster-safe splitter;
- `--min-free-gb 30` is active on every v2 writer, so curation refuses to flush
  a shard if disk headroom falls below the floor;
- training watcher:
  `scripts/watch_nonprotein_fot_then_launch_toricblm.py`.

The full-run config now points to the v2 manifest and v2 train globs, so the
launcher cannot accidentally train on incomplete v1 non-protein shards.

- `toricblm_nonprotein_fot_splits_v2_dna_*`
  - Script: `scripts/build_nonprotein_similarity_fot_splits.py`
  - Plan: `dna_coding_regions`.
  - Target: about 500k train rows.
  - Output: `data/toricblm_nonprotein_fot_splits/v2_parallel/dna`.
  - Split method:
    - sequence MinHash clusters plus organism/accession metadata.
  - FoT preservation: each emitted row includes `graph_json`, `forest_json`,
    `thought_forest_json`, `convextok_dag_json`, and `training_views_json`.

- `toricblm_nonprotein_fot_splits_v2_rna_*`
  - Plans: `rnacentral_8192` and `rfam_sequence`.
  - Target: about 500k train rows total across RNAcentral and Rfam.
  - Split method: sequence MinHash clusters plus family/type metadata.

- `toricblm_nonprotein_fot_splits_v2_small_*`
  - Plan: `pubchem_selfies`.
  - Target: about 500k train rows.
  - Split method: RDKit Murcko scaffold and Morgan fingerprint when SELFIES
    decodes; otherwise explicitly labeled SELFIES token-graph MinHash clusters
    for legacy/extended SELFIES rows.

- `toricblm_nonprotein_fot_watch_v2_*`
  - Writes a combined manifest every poll interval.
  - Starts `scripts/launch_toricblm_mup_full_codex55.sh` only when all three
    non-protein modalities have at least 450k train rows and sampled shards pass
    the FoT/schema checks.

- `toricblm_afdb_full_curation_3di_20260624T000435Z`
  - Replaced by `toricblm_afdb_full_curation_fot_20260624T034959Z` after the
    FoT-schema patch.
  - Existing written AFDB shards were upgraded in place with
    `scripts/upgrade_structure_fot_schema.py`.
  - Emits real AFDB coordinates, Foldseek/3Di strings where available, and
    complete graph/FoT/ConvexTok training-view sidecars.

- `toricblm_afdb_uniref50_curation_20260624T033307Z`
  - Replaced by `toricblm_afdb_uniref50_curation_fot_20260624T034959Z` after
    the FoT-schema patch.
  - New UniRef50 representative-accession AFDB curation.
  - Script support was repaired so UniRef50 rows can contribute accessions from
    `rep_accessions`, `rep_member_id`, `seed_id`, or `entry`.
  - Emits only real AFDB coordinate rows; no coordinate imputation; new shards
    include complete FoT sidecars.

## FoT Schema Repairs

- `scripts/build_afdb_structure_fot_dataset.py` now emits
  `thought_forest_json`, `convextok_dag_json`, `training_views_json`,
  `enrichment_status_json`, `leakage_signature_json`, and graph metadata for
  coordinate-native rows.
- `scripts/upgrade_structure_fot_schema.py` upgraded existing AFDB shards
  without changing real coordinate arrays.
- `scripts/repair_afdb_progress_from_shards.py` repaired interrupted AFDB
  progress files so accepted-but-unflushed accessions are retried after restart.
- `scripts/upgrade_authored_tot_fot_schema.py` upgraded authored Codex 5.5
  ToT graph shards, ProTrek v2 split shards, and PDB/PubChem3D coordinate graph
  shards to the same FoT sidecar format.
- A complete scan of the current `GRAPH_TRAIN_GLOB` matched 53 Parquet files
  and found zero FoT-schema errors after these repairs.

## Current Counts Snapshot

As of the v2-parallel transition:

- The old v1 non-protein FoT splits reached about 86k train rows but are
  superseded by v2 and are not referenced by the full-run config.
- A worker-thread smoke test for v2 DNA wrote 64 train rows plus held-out rows,
  validated the FoT schema, and exited cleanly after the pending-quota fix.
- The first v2 full launch used train-target early stopping.  It was stopped
  after only disposable first-minute shards because that mode can under-fill
  validation/test.  The active v2 launch fills train/validation/test quotas.
- A second restart added the writer-level disk guard after it passed a smoke
  test that filled train/validation/test with the same worker path.
- AFDB v6 full coordinate FoT: 13,312 train rows written so far.
- AFDB UniRef50 coordinate FoT: accepting rows after restart; no shard flushed
  yet because the shard size is 1,024 accepted coordinate rows.

## Training Relaunch Rule

Do not relaunch ToricBLM training until the non-protein manifest exists, the
minimum train-row counts pass, the FoT-format preflight passes, and the
structure-readiness preflight passes.  If a partial debugging launch is needed,
it must use a separate config with an explicit lower
`TORICBLM_NONPROTEIN_MIN_TRAIN_ROWS_PER_MODALITY` and must not be treated as the
clean full run.
