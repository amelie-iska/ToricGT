# ToricBLM FoT Tokenizer Audit

- Records/files audited: `['data/uniprot_fot/derived/toricblm_fot_graphified_512_per_dataset_leakage_v1.parquet']`
- Field-documents audited: `3072`
- Current tokenizer: `/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/parameter_golf_convextok8192_biomed_det_full/tokenizers/fineweb_convextok_8192_biomed_det.convextok.json`
- Comparison tokenizer: `/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/parameter_golf_convextok2048_det_full/tokenizers/fineweb_convextok_2048_det.convextok.json`

## Decision Notes

- Current ConvexTok bytes/token is 1.416, priced-token fraction is 0.261, and byte-fallback fraction is 0.739.
- Relative to ConvexTok-2048, ConvexTok-8192 token count delta is -7.21% on audited fields.
- The 8192 tokenizer improves coverage but not dramatically; keep it now and schedule a larger-vocab audit.
- Byte fallback is high; add a post-curation ConvexTok extension pass over graph_json, thought_forest_json, structure tags, GO/EC syntax, and sequence chunks.
- Sequence compression is weak; reserve more amino-acid/nucleotide motif tokens before large protein/dynamics training.

## Aggregate Metrics

| tokenizer | bytes/token | tokens/byte | priced fraction | byte fallback | reserve fraction |
|---|---:|---:|---:|---:|---:|
| current | 1.4161 | 0.7061 | 0.2608 | 0.7392 | 0.0592 |
| old | 1.3140 | 0.7610 | 0.2140 | 0.7860 | 0.0000 |

## Current Tokenizer By Field

| field | bytes/token mean | priced fraction mean | byte fallback mean | reserve fraction mean | active entropy mean |
|---|---:|---:|---:|---:|---:|
| enrichment_status_json | 1.8755 | 0.5104 | 0.4896 | 0.1162 | 0.0000 |
| graph_json | 1.4543 | 0.3109 | 0.6891 | 0.0607 | 0.0000 |
| sequence | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| text | 1.9054 | 0.5076 | 0.4924 | 0.1157 | 0.0000 |
| thought_forest_json | 1.6634 | 0.4069 | 0.5931 | 0.0686 | 0.0000 |
| training_views_json | 1.4751 | 0.3019 | 0.6981 | 0.0996 | 0.0000 |

## Training Implication

Use the existing ConvexTok-8192 tokenizer for the immediate full ToricBLM run if the aggregate fallback is acceptable. For future structure/dynamics phases, rerun this audit after adding coordinate, residue-pair, atom-type, chain-id, and trajectory metadata fields; only then decide whether 12k/16k vocabulary expansion is worth the parameter and artifact cost.
