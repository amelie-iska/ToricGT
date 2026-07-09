# Minimal Inference

This folder contains the smallest practical path for loading a ToricGT/ToricBLM
ConvexTok checkpoint from the Hub.  It expects a local checkout of the ToricGT
codebase because the model is still a research checkpoint with custom TokenGT,
ConvexTok, Forest-of-Thought, and tropical/toric graph modules.

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

Use `--flatten-graph-output` only for older BPB-flattening checkpoints.  The
current ToricBLM structure-priority runs are graph-in/graph-out and leave
flattening disabled.

Structure-flow coordinate export uses the ToricGT utility:

```bash
python scripts/export_structure_flow_trajectory.py \
  --frames path/to/generated_structure_frames.pt \
  --out-pdb outputs/generated_structure_trajectory.pdb
```

Checkpoints produced after the July 9, 2026 checkpoint patch include the
`structure_head` payload when structure-flow training is enabled.  Older
checkpoints can still run text/graph generation but may not contain the
structure-flow head state.
