# ToricBLM Bio Forest-of-Thought Dataset Plan

## Objective

Create a ToricGT/ToricBLM biological graph and Forest-of-Thought dataset from
the local raw Hugging Face biological-scale data at
`/home/iska/Documents/amelie/bio/iska-net/data/raw_hf_bio_scale`, with curated
outputs under `data/uniprot_fot` and publication to the Hugging Face dataset
repository `AmelieSchreiber/toricblm_fot`.

The dataset must serve the current ToricGT training paradigm:

- graph-in/graph-out records with explicit node and directed edge tokens;
- embedding-space Forest-of-Thought and GFlowNet trajectory training;
- GraphCG steerable-factor supervision over sequence/function/structure/taxonomy/
  chemistry axes;
- ConvexTok tokenization-DAG metadata compatible with tropical dynamic-programming
  token selection;
- tropical/toric attention diagnostics: active supports, margins, toric phase
  tags, one-dimensional cone terminology, and graph flattening views where needed;
- biomedical de novo design preparation without fabricating biological facts not
  present in the sources.

## Source Reality Check

The local raw data currently contains:

- `uniprot_function_text_train`: UniProt accession, entry name, protein name,
  protein sequence, and function text.
- `uniprot_uniref50_sequence_train`: UniRef50 clusters, representative sequence,
  taxonomy, GO MF/BP/CC lists, representative accessions, UniRef hierarchy, and
  hashes.
- `dna_coding_regions_train`: nucleotide sequence, organism, introns, exons, and
  translated protein features when present.
- `rfam_sequence_train`: RNA sequence, family, clan, and description.
- `rnacentral_8192_sequence_train`: RNAcentral UPI, sequence, type, description.
- `pubchem10m_selfies_train`: SELFIES molecular strings.

The local data does **not** contain full UniProtKB XML/JSON annotations for every
accession, full PDB mappings, AFDB coordinates, AlphaFold confidence values,
kinetic constants, sites, variants, pathways, or structure files. The curation
policy is therefore:

1. preserve every upstream field exactly when present;
2. parse only conservative labels such as GO ids and EC numbers when present;
3. add AFDB/PDB lookup nodes as external-enrichment hooks, not as fetched facts;
4. mark absent fields explicitly in `enrichment_status_json`;
5. do not synthesize biological claims.

## Research Basis

The dataset structure follows:

- Tree-of-Thought: represent intermediate reasoning as branchable thoughts with
  self-evaluation and backtracking/search.
- Graph-of-Thought: allow arbitrary dependency graphs, branch/merge operations,
  feedback, and thought aggregation.
- Forest-of-Thought: maintain several sparse reasoning trees, use dynamic
  self-correction, and choose leaves by consensus.
- ToricGT papers: treat records as typed graph-token data, expose edge tokens,
  reward graph/forest trajectories with GFlowNet-style trajectory balance, align
  hidden reasoning coordinates with GraphCG axes, and keep byte/ConvexTok views
  available for BPB-compatible flattening.

## Data Layout

`data/uniprot_fot/` will contain:

- `raw_sources/`: symlinks to raw data by default, avoiding a second 26GB copy.
  True move mode remains available through the builder but is not used unless
  explicitly needed.
- `derived/`: graphified/FoT Parquet and optional JSONL shards.
- `schema/`: JSON schema for curated rows.
- `manifests/`: raw scan and derived build manifests.
- `authored/`: hand-authored FoT/ToT records and source anchors.

## Record Columns

Each row should include the existing loader columns plus:

- `thought_forest_json`: explicit source-grounded FoT graph with several trees,
  expansion edges, correction edges, consensus node, budget levels, and trajectory
  balance metadata.
- `convextok_dag_json`: compact byte-boundary tokenization DAG for the training
  text prefix using the ConvexTok-8192 biomedical tokenizer when available.
- `training_views_json`: named views for graph-in/graph-out, forest trajectory,
  graph flattening, ConvexTok DAG scoring, and future design-conditioning tasks.
- `enrichment_status_json`: present/missing status for UniProtKB, GO, EC, sites,
  binding/catalytic sites, kinetic constants, PDB, AFDB, organism, taxonomy, RNA
  family, genomic features, and molecule string fields.

## Graphification Policy

For each source row:

1. Root node: source record with dataset, entry id, source shard, row index.
2. Sequence/string nodes:
   - biological sequence node when `sequence` exists;
   - SELFIES molecule node for PubChem rows;
   - sequence analytics node containing length, alphabet, composition, GC
     fraction for nucleotide/RNA, amino-acid composition for proteins, and
     conservative hash sketches;
   - chunk nodes for long sequences so TokenGT can reason over local segments.
3. Annotation nodes:
   - function/protein name/description/taxonomy/family/clan/source fields;
   - GO MF/BP/CC term nodes where present;
   - EC number nodes parsed from trusted text fields;
   - genomic feature nodes for exons/introns/proteins;
   - structure lookup nodes for AFDB/PDB when accession-like identifiers exist.
4. Directed edges:
   - `has_sequence`, `has_annotation`, `sequence_context_for_annotation`;
   - `has_go_annotation`, `has_ec_annotation`, `has_structure_lookup`;
   - `sequence_has_chunk`, `sequence_has_analytics`;
   - `annotation_supports_design_condition`, `structure_lookup_conditions_design`
     where applicable.
5. Targets:
   - source-grounded annotation bundle;
   - verifier metadata;
   - GFlowNet reward components;
   - active support nodes for tropical/TokenGT diagnostics.

## Forest-of-Thought Policy

Every graphified record receives an explicit FoT record:

- `sequence_tree`: sequence -> analytics -> chunks/features.
- `annotation_tree`: function/name/taxonomy/GO/EC/source labels.
- `structure_tree`: AFDB/PDB lookup hooks and structure-availability decisions.
- `design_condition_tree`: desired functional, structural, biochemical, and
  modality constraints grounded in available evidence.
- `chemistry_tree` for SELFIES records when applicable.
- `consensus` node that merges active leaves into a source-grounded annotation or
  design-conditioning target.
- `self_correction` edges that explicitly document decisions such as "structure
  absent locally, use lookup hook only" or "GO absent, do not hallucinate GO".

This is source-grounded deterministic FoT scaffolding, not imported LLM
rationale text. Hand-authored reasoning records can later add richer trajectories
in `data/uniprot_fot/authored`.

## ConvexTok / Tropical DP Policy

The ConvexTok tokenizer is itself a graph:

- vertices are byte boundaries;
- free edges are byte fallback tokens;
- priced edges are matched vocabulary tokens;
- a dynamic-programming path gives a rounded tokenization candidate;
- edge features include byte span, priced/free class, token id, byte length, and
  LP/rank scores when available.

For curated records we store a compact tokenization DAG for a bounded text prefix,
not a browser-computed or training-time expensive object.

## Disk Policy

The raw source directory is about 26GB and the filesystem has limited headroom.
Therefore:

- default mode is `symlink`, not copy;
- generated full records should be Parquet-first;
- JSONL should be optional and only used for small inspectable samples;
- full builds should shard by row count and write compressed Parquet.

## Implementation Checklist

- [x] Extend `scripts/build_uniprot_fot_dataset.py` schema with
  `thought_forest_json`, `convextok_dag_json`, `training_views_json`, and
  `enrichment_status_json`.
- [x] Add sequence/string analytics and conservative chunk nodes.
- [x] Add explicit source-grounded FoT trees with correction and consensus edges.
- [x] Add compact ConvexTok-8192 tokenization DAG extraction from the existing
  biomedical tokenizer.
- [x] Add Parquet-first sharded build options and optional JSONL output.
- [x] Update schema/docs/manifests.
- [x] Build and validate an initial curated shard set under `data/uniprot_fot`.
- [x] Publish the curated dataset artifacts to `AmelieSchreiber/toricblm_fot`.
- [x] Leave raw data as symlinks unless a later explicit move is required.

## Completed Initial Build

- Sample: `data/uniprot_fot/derived/toricblm_fot_graphified_sample_v1.{jsonl,parquet}`.
- Balanced Parquet slice: `data/uniprot_fot/derived/toricblm_fot_graphified_512_per_dataset_v1.parquet`.
- Records in balanced slice: 3,072 total, 512 from each of the six local raw datasets.
- Validation: both the sample JSONL/Parquet and the 3,072-record Parquet slice passed directed graph, FoT, ConvexTok-DAG, and training-view checks.
- Remote dataset: `https://huggingface.co/datasets/AmelieSchreiber/toricblm_fot`.
- Raw source handling: raw datasets are symlinked under `data/uniprot_fot/raw_sources`; no 26GB raw copy was made.

## Leakage-Aware Split Completion

The initial balanced slice now has an explicit leakage-aware split produced by
`scripts/apply_toricblm_fot_leakage_splits.py` and materialized by
`scripts/split_toricblm_fot_parquet.py`.

- Input: `data/uniprot_fot/derived/toricblm_fot_graphified_512_per_dataset_v1.parquet`.
- Leakage-enriched output: `data/uniprot_fot/derived/toricblm_fot_graphified_512_per_dataset_leakage_v1.parquet`.
- Split shards:
  - `data/uniprot_fot/splits/leakage_v1/toricblm_fot_leakage_v1_train.parquet`
  - `data/uniprot_fot/splits/leakage_v1/toricblm_fot_leakage_v1_validation.parquet`
  - `data/uniprot_fot/splits/leakage_v1/toricblm_fot_leakage_v1_test.parquet`
- Split counts: 2,788 train, 156 validation, 128 test.
- Cluster count: 2,640 leakage clusters; max cluster size 403.
- Split policy: deterministic cluster hash with approximately 90/5/5 train/validation/test allocation.

The split signatures combine sequence sketches, function labels/text shingles,
structure lookup hooks, source graph node/edge histograms, and FoT graph
topology.  This is a ProTrek/Foldseek-ready split: true trimodal contrastive
or structure-coordinate clustering should replace the hook-level structure
signature once coordinate-bearing PDB/AFDB/dynamics records are added.  The
current raw local data does not contain structure coordinates, so no Foldseek
or ProTrek structure-embedding inference has been claimed yet.

`external/ProTrek` is cloned as a lightweight code reference at commit
`4871546`.  Its README documents the future split upgrade: compute protein
sequence embeddings, Foldseek-derived structure embeddings, and function-text
embeddings, then cluster records by the maximum similarity across the three
modalities before assigning splits.  Model weights and structure-coordinate
assets were not downloaded in this pass; the current split therefore uses
source-grounded hook signatures rather than ProTrek inference outputs.

Validation passed for all three split shards:

```bash
python scripts/build_uniprot_fot_dataset.py validate \
  --jsonl data/uniprot_fot/splits/leakage_v1/toricblm_fot_leakage_v1_train.parquet
python scripts/build_uniprot_fot_dataset.py validate \
  --jsonl data/uniprot_fot/splits/leakage_v1/toricblm_fot_leakage_v1_validation.parquet
python scripts/build_uniprot_fot_dataset.py validate \
  --jsonl data/uniprot_fot/splits/leakage_v1/toricblm_fot_leakage_v1_test.parquet
```

## Tokenizer Decision

`scripts/analyze_toricblm_fot_tokenizer.py` audited the new graph/FoT records
against the current fresh ConvexTok-8192 biomedical tokenizer and the older
ConvexTok-2048 tokenizer.

- Current ConvexTok-8192 bytes/token: 1.416.
- Current priced-token fraction: 0.261.
- Current byte-fallback fraction: 0.739.
- ConvexTok-8192 reduced token count by 7.21% relative to ConvexTok-2048 on the
  audited graph/FoT fields.

Decision: keep ConvexTok-8192 for the immediate full run.  Do not retrain the
tokenizer before this run.  The high byte-fallback rate means a future
larger-vocabulary audit should target amino-acid motifs, nucleotide motifs,
GO/EC syntax, residue/atom/chain fields, coordinate labels, and trajectory tags
before large structure/dynamics training.

## Structure Training Status

`scripts/analyze_toricblm_structure_readiness.py` audited the train split:

- Records: 2,788.
- UniProt records: 916.
- Chemistry records: 505.
- Structure-association records: 917.
- Coordinate-bearing records: 0.
- Average graph size: 15.36 nodes, 20.61 edges.
- Average FoT size: 26.47 nodes, 44.75 edges.

The current run can train structure association through graph/FoT text, UniProt
function fields, GO/EC labels, sites/domains when present, SELFIES chemistry,
and AFDB/PDB lookup hooks.  It cannot honestly train coordinate-flow,
contact-map, or distogram losses yet because the local raw slice does not carry
coordinate tensors.  The new `src/toricgt/structure_flow_matching.py` module is
coordinate-native and ready for future coordinate-bearing records; those losses
remain dormant until such records are present.

## Full-Run Wiring

The late-stage graph stream is symlinked at:

```text
data/toricblm_late_mixed_fot_structure/train/*.parquet
```

It contains the existing Codex 5.5 ToT/FoT three-pass train files plus
`toricblm_fot_leakage_v1_train.parquet`.  The full training config is:

```text
configs/toricblm_mup_170m_convextok8192_biomed_fot_structure.env
```

Key choices:

- base FineWeb tokens: full ConvexTok-8192 tokenized dataset under the shared
  ToricGT data cache;
- model: muP 9-layer width-1536 target, about 170M parameters;
- graph LM and ToricGT sidecar: active every 4 steps;
- FoT/GFlowNet/MTP: active every 4 steps;
- late graph stream start: step 12,000;
- late graph mix ratio: 0.80;
- coordinate-native structure-flow weights are configured and logged, but
  coordinate losses are active only when coordinate-bearing records exist.
