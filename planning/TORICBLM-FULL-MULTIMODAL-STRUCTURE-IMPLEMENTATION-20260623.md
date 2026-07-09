# ToricBLM Full Multimodal Structure Implementation Plan - 2026-06-23

## Non-Negotiable Goal

The training path must move from a tiny AFDB seed shard to a real multimodal
biomedicine structure corpus.  The next training restart must not claim "all
modalities" unless the configured data streams actually contain those
modalities.  Missing coordinates must be skipped or reported as unavailable in
the coordinate-loss stream, not replaced by synthetic stand-ins or proxy
coordinate targets.

Structure availability is optional for the main graph/FoT training stream.  If
an entry has no corresponding PDB, AFDB, ESMFold, NDB, PubChem3D, or other
coordinate source, the entry must still be trained from every other available
field: sequence, annotations, function text, ontology labels, chemistry strings,
RNA/DNA/genomic features, Forest-of-Thought graph structure, ConvexTok DAG
features, TokenGT graph structure, leakage signatures, and enrichment status.
Only coordinate-specific losses skip that row.

## Current Truth

- The next active run is configured as graph-in/graph-out for the ToricBLM
  biological/FoT streams.  FineWeb sequence BPB remains logged for continuity,
  but the run disables graph-output flattening and adds explicit
  `graph_equiv/*` metrics from graph LM and complete-row graph losses.
- A trial graph-in/graph-out run was launched to validate integration and then
  stopped at step 0 after initial validation.  No optimizer steps were allowed
  to continue once the readiness gap was confirmed.
- Full training is now guarded by a launcher preflight:
  `TORICBLM_PREFLIGHT_READY_CHECK=1` plus
  `TORICBLM_REQUIRED_STRUCTURE_MODALITIES`.  The current manifest intentionally
  fails this gate because it contains AFDB protein coordinates but not yet PDB
  protein/RNA/DNA/complex or PubChem/ligand coordinate rows.
- PubChem, RNA, and DNA graph/FoT sequence or text rows are present in the
  late graph stream.  Coordinate-native PubChem/PDB/RNA/DNA rows are supported
  by the new builders and are included by glob when local shards exist.
- PDB/RCSB coordinate-native RNA/DNA/protein/complex curation is implemented
  with real mmCIF downloads and parser smoke tests.  Broad population remains a
  data job constrained by disk, not a synthetic fallback.
- Active prep jobs:
  - `toricblm_afdb_full_curation_3di_20260624T000435Z`: real AFDB coordinate
    shards with Foldseek/3Di saved per row.
  - RCSB/PDB modal curation completed train shards for protein, RNA, DNA, and
    protein-nucleic-acid/mixed complexes.  The broad query pass was stopped
    after verifying enough real rows for all required modalities, because the
    initial download-first behavior had poor observability and unnecessary disk
    use.
  - PubChem3D seed curation completed 88 real SDF conformer rows from curated
    biomedical CIDs.  The PubChem10M SELFIES shard uses an older SELFIES
    dialect with `[Branch]`/`[pop]` symbols rejected by the installed decoder,
    so it is not yet a reliable CID source without a dialect conversion layer.
- The strict readiness manifest now passes with train-coordinate counts:
  `protein_afdb=4335`, `pdb_protein=292`, `pdb_rna=10`, `pdb_dna=9`,
  `pdb_complex=59`, and `pubchem_3d_or_ligand=82`.
- ProTrek v2 splitting completed with actual GPU ProTrek inference:
  14,411 total rows, 13,227 train rows, 373 validation rows, 811 test rows,
  and 4,335 trimodal rows with sequence, text, and structure present.
- ProTrek and Foldseek are linked from the existing local ProTrek installation
  and smoke-tested on GPU.  The current ProTrek split was run on graph/FoT rows
  and produced 2,946 train, 76 validation, and 50 test rows over 55 clusters.
  The graph/FoT shard does not yet contain structure paths, so its
  structure-modality count is correctly reported as zero rather than inferred.
- A row-preserving full-entry graph/FoT training path is implemented for the
  OAI baseline adaptation.  It trains complete graphified rows up to 24,576
  ConvexTok tokens via full-row segmentation, next-segment contrastive
  prediction, boundary continuity, coverage anti-collapse, and sparse dense
  8,192-token full-context passes with activation checkpointing.

## Implementation Checklist

- [x] Replace the capped AFDB seed builder with a full sharded, resumable AFDB
  builder that scans all UniProt function rows, keeps long proteins by
  coordinate-window truncation, writes split Parquet shards, and deletes raw
  mmCIF files by default.
- [x] Ensure the all-entry graph/FoT curation path preserves entries without
  coordinate structures and annotates them with `structure_available=false`,
  `coordinate_training_available=false`, and an explicit
  `structure_missing_policy`.
- [x] Add generic coordinate-bearing structure record support for PDB/mmCIF
  structures so protein, RNA, DNA, protein-RNA, protein-DNA, RNA-ligand,
  DNA-ligand, and mixed biomolecular complexes can be emitted using real
  coordinates.
- [x] Add RCSB/PDB/NDB curation scripts that can fetch real mmCIF entries for
  RNA/DNA/protein/complex modalities and write the same coordinate schema.
- [x] Add PubChem coordinate ingestion only for real 3D conformer/SDF records
  when available.  Do not generate RDKit conformers as "real" structure data in
  this phase.
- [x] Extend `StructureCoordinateParquetStream` to accept multiple glob
  patterns and refresh shard lists, so long-running curation can feed new
  shards without hardcoding a single small file.
- [x] Configure `TORICBLM_STRUCTURE_TRAIN_GLOB` to point at all coordinate
  shard directories that are actually present and coordinate-bearing.
- [x] Add per-modality dataset manifests and readiness reports:
  protein-AFDB, PDB-protein, PDB-RNA, PDB-DNA, PDB-complex, PubChem-3D.
  The manifest reports only actually observed coordinate-bearing rows and
  fails when explicitly required modalities are absent.
- [x] Add ProTrek trimodal split support for protein-related data:
  sequence, structure, and function embeddings must be produced with ProTrek
  or the split report must fail when `--require-protrek` is passed.  No report
  may claim ProTrek-derived splitting unless ProTrek inference actually ran.
- [x] Add an explicit ProTrek readiness/check script and a trimodal splitting
  command that fails loudly when `external/ProTrek/weights/ProTrek_*` or the
  Foldseek binary/features are missing.  Do not emit a ProTrek split report
  from hash-only or metadata-only signatures.
- [x] Add optional flow-matching denoising trajectory export.  When the
  structure head generates or predicts a structure from context, inference
  should be able to request the intermediate denoising frames and write them as
  a multi-`MODEL` PDB and/or standard trajectory artifact.  This export must be
  labeled as a flow-matching generation path, not as a molecular dynamics
  trajectory; later dynamics training will use a separate physical trajectory
  schema with energies, forces, units, and integrator metadata.
- [x] Add a row-preserving complete-entry training path for large graph/FoT
  rows.  The active implementation uses `FullEntryGraphParquetStream`,
  segmented full-row local CE, next-segment InfoNCE, segment-boundary
  continuity, coverage anti-collapse, and sparse dense full-context passes.
- [x] Add a training restart config that uses the full multimodal coordinate
  globs and remains disk-safe.
- [x] Disable graph-output flattening for the new graph-in/graph-out run and
  report `graph_equiv/bpb`, `graph_equiv/primary_bpb`, and source counts so
  graph scoring is visible without pretending it is the OAI flattened sequence
  metric.
- [x] Add a full-training preflight gate so training cannot start while the
  manifest is missing declared required coordinate modalities.
- [x] Populate PDB protein, PDB RNA, PDB DNA, PDB mixed complex, and
  PubChem/ligand coordinate shards with real records, refresh the manifest, and
  rerun the preflight successfully before restarting a full run.
- [x] After coordinate population, run ProTrek trimodal splitting again over
  the combined train corpus so sequence, function text, saved Foldseek/3Di
  structure sequences, and graph/FoT signatures determine leakage clusters.
- [ ] Push the curated dataset to `AmelieSchreiber/toricblm_fot` only after the
  manifest is ready and the split report records actual ProTrek inference.
- [ ] Expand AFDB beyond the current shards as disk/time allow.  The curation
  job is still running and should not block training now that the manifest gate
  and ProTrek v2 split are ready.
- [ ] Add or select a SELFIES dialect conversion path if we want PubChem10M
  SELFIES-derived CIDs, because the current installed SELFIES decoder rejects
  the dataset's older `[Branch]`/`[pop]` syntax.

## Disk Policy

- Do not keep raw AFDB/PDB mmCIF files by default.
- Write compressed Parquet shards and JSON manifests.
- Keep only small smoke shards in git; full generated corpora should live in
  local data or Hugging Face dataset repos, not in GitHub.
- Before starting large curation, check free disk and refuse to run below a
  configured threshold.

## ProTrek Split Requirement

For protein-related records with sequence, structure, and text/function:

1. Produce sequence embeddings from amino-acid sequence.
2. Produce structure embeddings from Foldseek-derived 3Di sequence.
3. Produce text embeddings from function/annotation text.
4. Cluster by the joint trimodal signature, not by individual rows.
5. Assign clusters, not rows, to train/validation/test.
6. Record model path, weight hash, Foldseek binary path, number of records
   embedded in each modality, cluster count, max cluster size, and split counts.
7. If ProTrek weights or Foldseek are missing and `--require-protrek` is set,
   fail loudly.

## Restart Criteria

Restart training only after:

- code compiles;
- AFDB full builder smoke test emits multiple real coordinate shards;
- generic mmCIF parser smoke test emits at least one real coordinate row;
- structure loader smoke test reads multiple comma-separated globs;
- full-entry graph/FoT loader smoke test emits row-preserved large entries;
- one-step GPU smoke test exercises segmented full-row loss and sparse dense
  full-context loss with activation checkpointing;
- config points at coordinate-bearing shard directories;
- active run is stopped cleanly.
