# Epoch 1 Dataset Breakdown

Run: `toricblm-structure-priority-curriculum-20260709T175446Z`

Epoch 1 is the `structure_current` phase of the four-epoch structure-priority
curriculum.  The selected files are used for graph language modelling, full-row
long-entry training, and structure-flow coordinate training.

| field | value |
|---|---:|
| selected Parquet shards | 2,519 |
| selected rows | 5,033,975 |
| structure-flow coordinate rows | 5,033,975 |
| coverage steps | 2,516,988 |
| train batch tokens | 786,432 |
| grad accumulation | 8 |
| graph LM batch size | 6 |
| long-entry batch size | 2 |
| structure-flow batch size | 16 |
| long-entry max tokens per row | 25,600 |
| periodic full-context length | 16,384 |

## Rows By Modality

| modality | shards | rows |
|---|---:|---:|
| small molecule 3D | 2,202 | 4,503,122 |
| protein AFDB | 198 | 301,889 |
| PDB complex or structure | 119 | 228,964 |
| total | 2,519 | 5,033,975 |

## Pairing Contract

Coordinate-bearing rows remain in `GRAPH_TRAIN_GLOB`,
`LONG_ENTRY_TRAIN_GLOB`, and `TORICBLM_STRUCTURE_TRAIN_GLOB`.  That means
structure coordinates are trained together with the available graph JSON,
Forest-of-Thought JSON, ConvexTok DAG JSON, sequence or SELFIES fields, names,
labels, GO/EC/function annotations, tags, and PDB/PubChem/UniProt metadata
whenever those columns are present in the same row.

## Token Statistics From The Epoch Audit Sample

| statistic | ConvexTok tokens |
|---|---:|
| mean | 10,164.29 |
| p50 | 10,078 |
| p90 | 15,156 |
| p95 | 16,516 |
| p99 | 18,523 |
| max | 18,691 |

These statistics explain the current batch choices: full rows are preserved for
long-entry training, while the ordinary graph LM stream remains at `seq_len=1024`
and periodic full-context passes extend to `16,384` tokens.
