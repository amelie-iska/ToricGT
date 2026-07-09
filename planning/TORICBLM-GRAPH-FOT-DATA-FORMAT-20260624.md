# ToricBLM Graph/FoT Data Format For New Biological Data

Status: active implementation document, 2026-06-24.

## Goal

All newly curated biological rows are represented as graph-in/graph-out training
records with a Forest-of-Thought sidecar.  Raw upstream Hugging Face mirrors are
curation inputs only.  Training consumes split-safe Parquet shards whose rows
already contain graph structure, FoT structure, ConvexTok tokenization DAG
metadata, leakage signatures, and training-view metadata.

The main training objective remains graph-structured modeling.  Rows may still
be serialized for tokenizer scoring or auxiliary language losses, but the
canonical representation is the typed graph plus FoT tree/forest object.

## Canonical Row Columns

Every graph/FoT training row must include:

- `graph_json`: typed directed graph with nodes, edges, targets, metadata, and
  split-cluster information.
- `forest_json`: compact forest summary describing active evidence trees.
- `thought_forest_json`: canonical FoT object with schema
  `toricgt.biomed_source_grounded_forest_of_thought.v1`.
- `convextok_dag_json`: ConvexTok tokenization DAG with schema
  `toricgt.convextok_tokenization_dag.v1`.
- `training_views_json`: explicit mapping from the graph/FoT row into training
  views such as graph-in/graph-out, FoT/GFlowNet, GraphCG axes, tropical-toric
  fields, and optional structure-flow targets.
- `enrichment_status_json`: which biological evidence was present, computed,
  unavailable, or intentionally absent.
- `leakage_signature_json`: split and cluster provenance.
- `metadata_json`: row metadata used by analysis and training loaders.
- `split` and `split_cluster`: leakage-aware split assignment.

Structure-bearing rows also include real coordinate fields such as
`ca_coordinates`, `backbone_coordinates`, `structure_coordinates`,
`coordinate_mask`, residue or atom labels, and confidence fields.  Missing
structures are not fabricated; entries without structures remain trainable in
the graph/FoT stream using sequence, annotation, and source metadata.

## DNA Rows

DNA rows are built from local raw `dna_coding_regions_train` shards.  The graph
contains source nodes for the sequence, sequence analytics, chunked sequence
segments, coding-region annotations, exons/introns/protein links when present,
and source metadata.  Edges are directed and causal where possible: source
record to sequence, sequence to analytics, sequence to chunks, and sequence to
annotations.

The FoT structure has evidence trees for sequence, genomic annotations, design
conditions, and missing-evidence self-corrections.  Long sequences are chunked
with a high chunk budget so full rows remain available to long-entry training.

Leakage control uses sequence MinHash clusters plus available metadata such as
organism, type, family, and accession prefix.  Split assignment is by cluster,
not by individual row.

## RNA Rows

RNA rows are built from RNAcentral and Rfam sequence shards.  They follow the
same graph/FoT structure as DNA rows, with RNA-specific source fields such as
family, clan, type, description, accession/UPI, and sequence analytics.

RNA downsampling is split across RNAcentral and Rfam source plans.  The current
target is roughly 500k train rows total for RNA, with validation and test rows
assigned by the same sequence-cluster policy.

## Small-Molecule Rows

Small-molecule rows are built from PubChem SELFIES shards.

When SELFIES can be decoded into a molecule, the split signature uses RDKit
Murcko scaffold and Morgan-fingerprint locality buckets.  When a row uses a
legacy or extended SELFIES alphabet that the local decoder cannot parse, the row
is still kept if it has a valid SELFIES token sequence; in that case the split
signature is explicitly labeled as a SELFIES token-graph MinHash cluster, not as
a chemical scaffold.  This avoids pretending that token-graph evidence is a
chemical scaffold while still retaining useful molecule-string training rows.

The graph exposes SELFIES token nodes, source molecular string metadata,
ConvexTok DAG features, and future design-condition fields.  Real 3D PubChem
conformer rows, when available from separate structure curation, carry coordinate
columns and structure-flow targets.

The current target is roughly 500k train rows for small molecules, selected as
a diverse subset rather than a near-duplicate stream.

## Protein And Structure Rows

Protein text/function rows use ProTrek-derived split shards where available.
Protein rows with structures use real AFDB/PDB coordinate curation.  Structure
rows are coordinate-native graph/FoT records:

- graph nodes represent protein/sequence/function/structure/coordinate targets
  and residue or atom blocks;
- edges represent sequence-to-structure, structure-to-coordinate-target,
  residue-block adjacency, and contact-block relationships;
- FoT trees represent sequence evidence, geometry evidence, confidence and
  uncertainty, function evidence, design constraints, and self-corrections;
- flow-matching training consumes the real coordinate columns, not synthetic
  substitutes.

AFDB curation preserves Foldseek/3Di strings when available.  UniRef50
representative accessions are handled by parsing accessions from `rep_accessions`,
`rep_member_id`, `seed_id`, or `entry`.

## Downsampling Policy

The non-protein goal is a diverse subset that roughly doubles the current
protein-derived training scale per modality: about 500k train rows each for DNA,
RNA, and small molecules.  This is enforced by:

- split assignment by cluster, not by row;
- a per-cluster cap to prevent high-copy families or repeated molecules from
  dominating;
- separate train/validation/test assignment by cluster hash;
- preserving all trainable evidence for rows that pass the diversity filter.

The target is configurable.  The full-run config currently requires at least
450k train rows per non-protein modality before the clean training launcher will
start.

## Parallel Curation

The curation script now supports plan-level selection and bounded graphify
workers:

```bash
python scripts/build_nonprotein_similarity_fot_splits.py \
  --out-dir data/toricblm_nonprotein_fot_splits/v2_parallel \
  --include-plan dna_coding_regions \
  --num-workers 10
```

Split assignment and cluster caps stay in the parent process.  Worker threads
materialize the expensive graph/FoT/ConvexTok row payloads, while separate tmux
processes run DNA, RNA, and small-molecule plans in parallel.  This keeps
leakage logic centralized per modality while using the CPU effectively without
forking large tokenizer payloads.

The final manifest is produced by
`scripts/finalize_nonprotein_fot_manifest.py`, and the training watcher is
`scripts/watch_nonprotein_fot_then_launch_toricblm.py`.

## Training Consumption

The clean full-run config points `GRAPH_TRAIN_GLOB` and `LONG_ENTRY_TRAIN_GLOB`
only at curated split-safe graph/FoT shards.  It refuses to train if raw
`raw_hf_bio_scale` paths appear in those globs.

The launcher additionally checks sampled graph shards for the required FoT
columns and schemas.  This prevents a mixed run where some data are graph/FoT
records and others are plain rows.

## Non-Negotiable Invariants

- Do not fabricate missing structures.
- Do not use raw upstream train shards directly for model training.
- Do not call token-graph SELFIES clusters chemical scaffolds.
- Preserve FoT formatting for all rows used by graph/FoT training.
- Keep split assignment cluster-based for leakage resistance.
- Keep graph-in/graph-out as the canonical training representation.
