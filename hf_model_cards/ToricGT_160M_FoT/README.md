---
license: apache-2.0
library_name: pytorch
pipeline_tag: text-generation
tags:
  - toricgt
  - toricblm
  - forest-of-thought
  - gflownet
  - tokengt
  - convextok
  - tropical-geometry
  - toric-geometry
  - graph-transformer
  - biomedical-reasoning
  - parameter-golf
datasets:
  - HuggingFaceFW/fineweb
  - AmelieSchreiber/toricgt-curated-splits
  - AmelieSchreiber/codex5.5_ToT
base_model:
  - openai/parameter-golf
---

# ToricGT 170M ConvexTok-8192 FoT

<p>
  <a href="https://amelie-iska.github.io/ToricGT/"><img alt="Project page" src="https://img.shields.io/badge/project-ToricGT-38d6ff?style=for-the-badge"></a>
  <a href="https://github.com/amelie-iska/ToricGT/tree/toricblm"><img alt="GitHub" src="https://img.shields.io/badge/github-code-111827?style=for-the-badge&logo=github"></a>
  <a href="https://huggingface.co/collections/AmelieSchreiber/toricgt"><img alt="ToricGT collection" src="https://img.shields.io/badge/HF_collection-ToricGT-ffcc4d?style=for-the-badge"></a>
  <a href="https://huggingface.co/blog/AmelieSchreiber/toricblm"><img alt="ToricBLM blog" src="https://img.shields.io/badge/blog-ToricBLM-ff69b4?style=for-the-badge"></a>
</p>

ToricGT 170M ConvexTok-8192 FoT is the current muP-scaled ToricGT/ToricBLM checkpoint line: a width-scaled adaptation of the OpenAI Parameter Golf baseline with first-class graph-structured training, fresh ConvexTok tokenization, TokenGT-style graph tokenization, graph-output flattening for BPB scoring, embedding-space Forest-of-Thought reasoning, GFlowNet-style trajectory objectives, GraphCG disentanglement, and tropical-toric/topological/category-theoretic audit losses.

This repository is prepared while the ConvexTok-8192 ToricBLM run is being exported/launched.  The model card will be updated with dated checkpoint artifacts after the run produces checkpoints.

## Current Training Run

| Item | Value |
|---|---|
| Run id | `toricblm-mup-codex8192-biomed-full-<timestamp>` |
| Approximate parameters | `169,698,927` generated target params |
| Width transfer | muP base 512, delta 768, target 1536, width multiplier 3.0 |
| Layers | 9 |
| Heads / KV heads | 12 / 6 |
| Sequence length | 1024 |
| Effective batch tokens | 196,608 initial setting for larger softmax |
| Tokenizer | fresh ConvexTok deterministic 8192, no BPE, no old-vocab reuse |
| Biomedical/control reserve | 512-token target reserve from universal-modality biomedical seed syntax |
| Primary BPB stream | FineWeb-style ConvexTok-8192 shards exported from ToricGT curated Parquet source |
| Graph stream | full `AmelieSchreiber/toricgt-curated-splits` train split |
| Late-stage graph stream | `AmelieSchreiber/codex5.5_ToT`, train-only, 3-pass view |
| Late-stage mix | 85% codex5.5_ToT / FoT graph stream after step 17,500 |
| Validation | FineWeb BPB validation stream |

## What Is Being Trained

The run combines five layers of structure:

1. **FineWeb BPB objective.**  Bits-per-byte remains the score-bearing language-model objective.
2. **ConvexTok DAG tokenization.**  Tokenization is represented as a byte-boundary DAG: vertices are byte boundaries, candidate arcs are tokens, rounded/frequency-ranked candidate features become token/edge features, and the final flattened sequence remains compatible with BPB scoring.  The active scale-up tokenizer is a fresh ConvexTok-8192 vocabulary with a biomedical/control reserve for future UniProt, PDB/AFDB/ESMFold, molecular-graph, atomistic-trajectory, cell-state, tropical-toric, persistence, category-theoretic, and FoT control streams.
3. **TokenGT graph input/output.**  Text and graph records are graphified, with node tokens, edge tokens, endpoint structure, positional buckets, toric phase features, and optional graph-output flattening for OAI FineWeb scoring.
4. **Embedding-space FoT and GFlowNets.**  The model trains branching reasoning trajectories in embedding space with Forest-of-Thought style exploration, GFlowNet rewards, trajectory memory retrieval, and analogical retrieval heads.
5. **Mathematical audits and regularizers.**  Tropical attention, toric embeddings, one-dimensional-cone/vector-bundle probes, BGG category O certificates, Koszul/persistence metrics, combinatorial commutative algebra, derived signatures, and GraphCG basis-vector disentanglement are active as low-weight regularizers and W&B metrics.

## Data

The model uses:

- Fresh ConvexTok-8192 shards for the primary BPB objective, exported from the existing ToricGT local data root.
- [`AmelieSchreiber/toricgt-curated-splits`](https://huggingface.co/datasets/AmelieSchreiber/toricgt-curated-splits) train split as the full curated graph training stream:
  - train: 4,633,582 rows, 117 parquet shards
  - validation/test are kept for evaluation and are not used for graph training.
- [`AmelieSchreiber/codex5.5_ToT`](https://huggingface.co/datasets/AmelieSchreiber/codex5.5_ToT) as a late-stage train-only Tree/Forest-of-Thought dataset:
  - 3,890 source rows
  - materialized as a deterministic three-pass late-stage view
  - no validation/test split is used from this dataset.

## Running a Checkpoint

This is not a standard `transformers` model repo.  The checkpoint is loaded through the ToricGT codebase and the adapted OpenAI Parameter Golf baseline.

```bash
git clone --recursive https://github.com/amelie-iska/ToricGT.git
cd ToricGT
git checkout toricblm

# After a dated checkpoint is uploaded here:
hf download AmelieSchreiber/ToricGT_160M_FoT \
  --include "checkpoints/*_step_*.pt" \
  --local-dir ./hf_downloads/ToricGT_160M_FoT

# Example evaluation entry point from the ToricGT repo.
conda run -n tokengt env PYTHONPATH=src:external/mup:amelie-iska/parameter-golf \
  python scripts/evaluate_oai_competition_bpb.py \
  --checkpoint ./hf_downloads/ToricGT_160M_FoT/checkpoints/<dated-checkpoint>.pt \
  --tokenizer /home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/parameter_golf_convextok8192_biomed_det_full/tokenizers/fineweb_convextok_8192_biomed_det.convextok.json
```

The final model artifacts uploaded here will include the dated checkpoint, training config, muP base-shape file, run log, and supporting planning notes.

## Relation to ToricBLM

This checkpoint line is intended as the scale-up bridge from ToricGT toward ToricBLM: a biomedical de novo design and scientific reasoning model built around universal equivariant graph-to-graph approximation, tropical-toric reasoning control, analogical memory retrieval, and Forest-of-Thought inference-time scaling.  The model is trained for general graph-structured scientific reasoning and does not provide medical advice or patient-specific recommendations.

## Limitations

- This is an experimental research checkpoint line.
- The current 170M run is outside the 16MB OpenAI Parameter Golf submission regime.
- Mathematical auxiliary objectives are low-weight training signals and audits; they should not be interpreted as formal proof that every generated reasoning trajectory is algebraically valid.
- Biomedical use requires separate safety evaluation and domain-specific validation.

## Links

- Project page: https://amelie-iska.github.io/ToricGT/
- GitHub code: https://github.com/amelie-iska/ToricGT/tree/toricblm
- ToricGT collection: https://huggingface.co/collections/AmelieSchreiber/toricgt
- ToricBLM blog: https://huggingface.co/blog/AmelieSchreiber/toricblm
- Curated graph dataset: https://huggingface.co/datasets/AmelieSchreiber/toricgt-curated-splits
- Late-stage ToT/FoT dataset: https://huggingface.co/datasets/AmelieSchreiber/codex5.5_ToT
- Smaller checkpoint line: https://huggingface.co/AmelieSchreiber/toricgt-checkpoints
