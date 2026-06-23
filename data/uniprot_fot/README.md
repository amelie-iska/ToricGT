# UniProt FoT Graphification Workspace

This directory is the ToricGT landing area for biological graph/FoT training
data derived from `/home/iska/Documents/amelie/bio/iska-net/data/raw_hf_bio_scale`.

The first implementation pass is conservative because the source data is about
26GB and the host filesystem is nearly full. Raw datasets are organized here by
symlink under `raw_sources/` by default. This keeps the original data reachable
from the ToricGT data tree without duplicating it. A true `move` is supported by
the build script, but should only be used after confirming that no `iska-net`
workflow depends on the current raw path.

## Record Families

- `uniprot_function_text_train`: protein sequence plus function and protein-name
  text, with EC labels extracted when present and AlphaFold DB lookup URLs when
  accessions allow them.
- `uniprot_uniref50_sequence_train`: UniRef50 clusters with sequence, taxonomy,
  GO MF/BP/CC lists, representative accessions, and cluster metadata.
- `rfam_sequence_train`: RNA family/clan sequence annotations.
- `rnacentral_8192_sequence_train`: RNAcentral sequence/type/description rows.
- `dna_coding_regions_train`: genomic sequence, exons, introns, translated
  proteins, organism, and accession fields.
- `pubchem10m_selfies_train`: SELFIES molecular strings as graph-tokenizable
  medicinal-chemistry strings.

## Graph/FoT Format

Each derived row has flat Parquet fields for training loaders plus a full
`graph_json` object with:

- `id`, `source`, `task_family`, `nodes`, `edges`, `targets`, `metadata`, and
  `split_cluster`.
- Directed causal edges between source record, sequence, annotation, structure
  lookup, GO, EC, and feature nodes.
- Stable hash-based latent coordinate proxies for continuous/hybrid GFlowNet
  metadata. These are curation features, not learned embeddings.
- TokenGT/TropicalGT/ToricGT metadata: node token order, edge token order,
  tropical active support nodes, a tropical margin proxy, toric phase-basis
  tags, and ConvexTok/byte-packing compatibility notes.
- GFlowNet reward metadata based on source-field density, sequence presence,
  GO/EC/structure availability, and directed graph connectivity.
- Leakage-resistant split clusters based on dataset, entry/accession, sequence
  hashes, or sequence prefixes.

The companion `thought_forest_json` is the stronger training view. It
materializes source-grounded Forest-of-Thought nodes and edges with sparse tree
activation, expansion, self-correction, and consensus. The trees are:

- `sequence_tree`: source sequence, deterministic sequence analytics, and local
  sequence chunks.
- `annotation_tree`: function text, protein names, taxonomy, GO, EC, family,
  clan, and other source annotations.
- `structure_tree`: PDB/AFDB lookup hooks when accession-like identifiers permit
  them; otherwise an explicit self-correction edge to enrichment status.
- `design_condition_tree`: future conditional design constraints grounded in
  available source evidence.
- `chemistry_tree`: SELFIES token evidence for molecule rows.

This is source-grounded deterministic FoT scaffolding, not imported LLM
rationales and not hallucinated biological knowledge. If full UniProtKB fields,
PDB mappings, AFDB coordinates, sites, binding sites, catalytic activity, or
kinetic constants are absent from the local row, `enrichment_status_json` records
that absence and the forest routes through a self-correction edge instead of
inventing the missing evidence.

The `convextok_dag_json` field stores a compact ConvexTok-8192 biomedical
tokenization graph over a bounded text prefix: byte-boundary vertices, free byte
fallback edges, matched priced vocabulary edges, and an exact min-plus
shortest-path dynamic program. This is the tokenizer-side graph that aligns the
bio records with ToricGT's tropical-attention/TokenGT training path.

The `training_views_json` field exposes named views for graph-in/graph-out,
embedding-space FoT/GFlowNet training, ConvexTok flattening, GraphCG axes, and
tropical-toric active support metadata. The older `forest_json` field is kept as
a backward-compatible compact source-field tree summary.

## Build And Validate

Run from the ToricGT repository root on branch `toricblm-data`:

```bash
/home/iska/miniconda3/envs/iska-net-2/bin/python scripts/build_uniprot_fot_dataset.py build \
  --raw-root /home/iska/Documents/amelie/bio/iska-net/data/raw_hf_bio_scale \
  --output-dir data/uniprot_fot \
  --sample-per-dataset 8 \
  --raw-link-mode symlink

/home/iska/miniconda3/envs/iska-net-2/bin/python scripts/build_uniprot_fot_dataset.py validate \
  --jsonl data/uniprot_fot/derived/uniprot_fot_graphified_sample.jsonl
```

For larger local builds, prefer Parquet-only output:

```bash
/home/iska/miniconda3/envs/iska-net-2/bin/python scripts/build_uniprot_fot_dataset.py build \
  --raw-root /home/iska/Documents/amelie/bio/iska-net/data/raw_hf_bio_scale \
  --output-dir data/uniprot_fot \
  --records-per-dataset 512 \
  --output-prefix toricblm_fot_graphified_512_per_dataset \
  --raw-link-mode symlink \
  --no-jsonl
```

Outputs:

- `raw_sources/`: symlink organization layer for the original raw datasets.
- `manifests/uniprot_fot_build_manifest.json`: raw and derived data manifest.
- `derived/uniprot_fot_graphified_sample.jsonl`: inspectable JSONL rows.
- `derived/uniprot_fot_graphified_sample.parquet`: Parquet training sample.

## Leakage Split, Tokenizer Audit, And Structure Readiness

The current balanced build is:

```text
derived/toricblm_fot_graphified_512_per_dataset_v1.parquet
```

It has 3,072 records, with 512 from each local raw source family.  The
leakage-aware splitter writes:

```text
derived/toricblm_fot_graphified_512_per_dataset_leakage_v1.parquet
splits/leakage_v1/toricblm_fot_leakage_v1_train.parquet
splits/leakage_v1/toricblm_fot_leakage_v1_validation.parquet
splits/leakage_v1/toricblm_fot_leakage_v1_test.parquet
```

Split counts are 2,788 train, 156 validation, and 128 test.  Clusters combine
sequence sketches, function/annotation shingles, structure lookup hooks, source
graph histograms, and FoT graph topology.  This is a leakage guard for the data
available now; coordinate-level Foldseek or ProTrek trimodal clustering should
be applied after coordinate-bearing PDB/AFDB/dynamics records are added.

The ConvexTok audit is recorded in
`../../planning/TORICBLM-FOT-TOKENIZER-AUDIT.md`.  The current decision is to
keep the fresh ConvexTok-8192 biomedical tokenizer for the immediate full run
and revisit vocabulary extension when residue, atom, coordinate, trajectory,
and richer UniProt/PDB fields are included.

`manifests/toricblm_fot_structure_readiness_train_v1.json` reports 917
structure-association records and zero coordinate-bearing records in the train
split.  Structure association is therefore trainable now through graph/FoT
annotations and lookup hooks, while coordinate-native flow/contact/distogram
losses remain dormant until actual coordinate tensors are curated.

The late-stage training glob used by the full ToricBLM run is symlink-only:

```text
../toricblm_late_mixed_fot_structure/train/*.parquet
```

It combines the existing Codex 5.5 ToT/FoT three-pass train files with
`splits/leakage_v1/toricblm_fot_leakage_v1_train.parquet`.

## Next Dataset Layer

The next layer should be a separate authored FoT/ToT corpus, not a generator
dump. Each record should be written as a technical reasoning artifact grounded
in one or more source rows, especially UniProt/UniRef records with rich
functional, GO, EC, site, family, pathway, structure, perturbation, and design
constraints. Scripts may validate, hash, shard, and publish accepted records,
but authored reasoning text should be inspected before acceptance.
