# ToricBLM FoT + Structure-Association Full Run Plan

Date: 2026-06-23

## Goal

Launch a full ToricBLM training run that keeps the OpenAI Parameter-Golf BPB
objective primary, trains the 170M muP ConvexTok-8192 model on the full
FineWeb tokenized corpus, and introduces late-stage graph/FoT biological
records plus Codex 5.5 ToT/FoT records for reasoning, analogical memory, and
biomedicine preparation.

## Inputs

- Main BPB dataset:
  `/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/parameter_golf_convextok8192_biomed_det_full/datasets/fineweb10B_convextok8192_biomed_det`
- Tokenizer:
  `/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/parameter_golf_convextok8192_biomed_det_full/tokenizers/fineweb_convextok_8192_biomed_det.convextok.json`
- Late mixed graph stream:
  `data/toricblm_late_mixed_fot_structure/train/*.parquet`
- New ToricBLM bio/FoT leakage-aware split:
  `data/uniprot_fot/splits/leakage_v1/toricblm_fot_leakage_v1_train.parquet`
- Structure readiness manifest:
  `data/uniprot_fot/manifests/toricblm_fot_structure_readiness_train_v1.json`
- Config:
  `configs/toricblm_mup_170m_convextok8192_biomed_fot_structure.env`

## Tokenizer Decision

The ConvexTok-8192 audit shows a 7.21% token-count reduction relative to
ConvexTok-2048 on the audited graph/FoT fields, but a high byte fallback rate
for raw sequence fields.  The decision for this run is to keep ConvexTok-8192
rather than retraining immediately.  A later tokenizer extension should use
coordinate-bearing structure, dynamics, residue-pair, atom-type, chain-id,
GO/EC, amino-acid motif, nucleotide motif, and trajectory fields.

## Leakage Split

The new train/validation/test split is cluster-based rather than random.  Its
signatures combine sequence sketches, function labels/text shingles, structure
lookup hooks, source graph histograms, and FoT graph topology.  This prevents
direct leakage of near-duplicate sequence/function/FoT structures across splits
for the current data.  True ProTrek trimodal or Foldseek structural clustering
is deferred until coordinate-bearing structure records are present.

## Structure Loss Status

The current curated train split contains 917 structure-association records but
zero coordinate-bearing records.  Therefore:

- graph/FoT structure association is active through the graph LM and ToricGT
  sidecar;
- coordinate-native flow matching, contact-map BCE, distogram CE, and frame/RMSD
  losses are implemented in `src/toricgt/structure_flow_matching.py`;
- those coordinate losses are logged as configured but dormant until real
  coordinate tensors are added to training records.

This avoids proxy coordinate targets and keeps the implementation ready for
future PDB/AFDB/UMA/BioEmu/BioKinema/ConfRover trajectory phases.

## Training Configuration

Key settings in
`configs/toricblm_mup_170m_convextok8192_biomed_fot_structure.env`:

- `MODEL_DIM=1536`, `NUM_LAYERS=9`, `NUM_HEADS=12`, `NUM_KV_HEADS=6`.
- `VOCAB_SIZE=8192`, tied embeddings, muP base-shape transfer enabled.
- `TRAIN_SEQ_LEN=1024`, `TRAIN_BATCH_TOKENS=196608`, `ITERATIONS=25000`.
- First-class FineWeb TokenGT graphification and ConvexTok DAG features on.
- Graph-output flattening active only for OAI FineWeb BPB scoring.
- Graph LM and ToricGT sidecar active every 4 steps.
- OAI GFlowNet, embedding-space FoT, and MTP active every 4 steps.
- Late graph stream starts at step 12,000 with mix ratio 0.80.
- Structure readiness and coordinate-loss configuration are reported to W&B
  under `toricblm_structure/*`.

## Launch Command

```bash
CONFIG_PATH=/home/iska/Documents/amelie/bio/ToricGT/configs/toricblm_mup_170m_convextok8192_biomed_fot_structure.env \
RUN_ID=toricblm-mup-fot-structure-$(date -u +%Y%m%dT%H%M%SZ) \
./scripts/launch_toricblm_mup_full_convextok8192_biomed.sh
```

The run should be launched in `tmux` so it continues after the shell returns.

## Review Criteria

Monitor:

- FineWeb train/validation BPB and the 1K validation cadence;
- graph LM late-stream activation at step 12K;
- `oai_fot/*`, `oai_gflownet/*`, and `oai_mtp/*` for reasoning-search stability;
- `toricgt_sidecar/*` for GraphCG, analogy, memory, toric, BGG, Koszul, derived,
  and vector-bundle 1D-cone signals;
- `toricblm_structure/*` to confirm coordinate losses remain dormant until
  coordinate-bearing records are added;
- tokenizer toric/tropical metrics for any evidence that the 8192 vocab is
  under-serving biological strings.
