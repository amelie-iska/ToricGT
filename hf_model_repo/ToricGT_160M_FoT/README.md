---
license: apache-2.0
language:
  - en
tags:
  - toricgt
  - toricblm
  - graph-transformer
  - forest-of-thought
  - convextok
  - protein-design
  - structure-generation
library_name: pytorch
pipeline_tag: text-generation
---

# ToricGT / ToricBLM 160M FoT Checkpoints

[Project page](https://amelie-iska.github.io/ToricGT/) ·
[GitHub](https://github.com/amelie-iska/ToricGT) ·
[ToricGT collection](https://huggingface.co/collections/AmelieSchreiber/toricgt) ·
[ToricBLM blog](https://huggingface.co/blog/AmelieSchreiber/toricblm)

This repository stores ToricGT/ToricBLM research checkpoints and the minimum
supporting artifacts needed to run the current 160M-parameter ConvexTok,
TokenGT-style, graph-in/graph-out Forest-of-Thought model.  The active training
line uses:

- ConvexTok-8192 biomedical byte-exact tokenization with tokenization-DAG
  features.
- TokenGT-style graphification for input and output, with ordinary BPB no longer
  using graph-output flattening in the current structure-priority run.
- Embedding-space Forest-of-Thought and GFlowNet heads.
- Full-rank GraphCG-style basis disentanglement.
- Tropical/toric, persistent-homology, Koszul, BGG category O, vector-bundle
  one-dimensional-cone, sheaf, and derived-signature sidecar losses and metrics.
- Structure-flow coordinate training for multimodal protein, PDB/complex, and
  small-molecule structure rows.

## Current Structure-Priority Run

Active run: `toricblm-structure-priority-curriculum-20260709T175446Z`

Epoch 1 is structure-first and trains on all currently selected coordinate rows:

| modality | shards | rows |
|---|---:|---:|
| small molecule 3D | 2,202 | 4,503,122 |
| protein AFDB | 198 | 301,889 |
| PDB complex or structure | 119 | 228,964 |
| total | 2,519 | 5,033,975 |

See [`epoch_001_dataset_breakdown.md`](epoch_001_dataset_breakdown.md) for the
full breakdown.

## Minimum Needed To Run

Clone ToricGT, install the minimal Python stack, then run the loader in
`inference/`.

```bash
git clone https://github.com/amelie-iska/ToricGT.git
cd ToricGT
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy huggingface_hub

python hf_model_repo/ToricGT_160M_FoT/inference/minimal_toricblm_generate.py \
  --repo-id AmelieSchreiber/ToricGT_160M_FoT \
  --checkpoint checkpoints/<run-id>/<checkpoint-file>.pt \
  --prompt-file hf_model_repo/ToricGT_160M_FoT/examples/enzyme_design_fot_structure_prompt.json \
  --max-new-tokens 768
```

The default tokenizer path is
`tokenizers/fineweb_convextok_8192_biomed_det.convextok.json`, which is included
in this model repository.

Use `--flatten-graph-output` only with older sequence-flattening BPB checkpoints.
The current ToricBLM structure-priority run is graph-in/graph-out.

## Example Prompts

Two ready-to-run prompts are included:

- [`examples/enzyme_design_fot_structure_prompt.json`](examples/enzyme_design_fot_structure_prompt.json):
  a Forest-of-Thought prompt to design three enzyme classes with kinetic targets
  and structure-flow export requests, including a 250-400 residue PETase-like
  hydrolase.
- [`examples/de_novo_binders_difficult_target_prompt.json`](examples/de_novo_binders_difficult_target_prompt.json):
  a Forest-of-Thought prompt to generate 10 diverse de novo protein binders to
  KRAS G12D, with interface, specificity, and structure-flow requests.

These examples are computational-design prompts.  They intentionally ask for
graph/FoT reasoning artifacts, verification tables, and structure-flow export
requests rather than wet-lab protocols.

## Structure-Flow Output

The trainer-side structure-flow head predicts coordinate denoising targets.
Checkpoints produced after the July 9, 2026 checkpoint patch include
`structure_head` when structure-flow training is enabled.  Older checkpoints can
still run graph/text inference but may not contain that head.

To export a generated coordinate trajectory as multi-MODEL PDB:

```bash
python scripts/export_structure_flow_trajectory.py \
  --frames path/to/generated_structure_frames.pt \
  --out-pdb outputs/generated_structure_trajectory.pdb
```

## Checkpoints

Epoch-special checkpoints from the structure-priority curriculum are uploaded
under:

```text
checkpoints/<run-id>/<checkpoint-file>.pt
```

The training watchdog uploads each special epoch checkpoint as soon as it is
saved, along with the epoch manifest and the corresponding dataset-state repo.
