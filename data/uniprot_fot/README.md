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

The companion `forest_json` organizes each source row into four deterministic
source-field trees: sequence, annotation, structure lookup, and future design
conditions. This is not the authored reasoning dataset yet; it is the graphified
raw-data substrate that later authored FoT/ToT trajectories can cite.

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

Outputs:

- `raw_sources/`: symlink organization layer for the original raw datasets.
- `manifests/uniprot_fot_build_manifest.json`: raw and derived data manifest.
- `derived/uniprot_fot_graphified_sample.jsonl`: inspectable JSONL rows.
- `derived/uniprot_fot_graphified_sample.parquet`: Parquet training sample.

## Next Dataset Layer

The next layer should be a separate authored FoT/ToT corpus, not a generator
dump. Each record should be written as a technical reasoning artifact grounded
in one or more source rows, especially UniProt/UniRef records with rich
functional, GO, EC, site, family, pathway, structure, perturbation, and design
constraints. Scripts may validate, hash, shard, and publish accepted records,
but authored reasoning text should be inspected before acceptance.
