# ToricGT Hub Data And Weights

This directory is the local landing area for ToricGT dataset splits and release
metadata. Large Parquet files, raw downloads, W&B logs, and checkpoints are
ignored by git; this README is tracked so the Hub locations are visible from the
repository.

## Hugging Face Collection

- Public collection URL: <https://huggingface.co/collections/AmelieSchreiber/toricgt>
- Resolved collection slug: `AmelieSchreiber/toricgt-6a15263a63ee47022934faa2`
- Title: `ToricGT`
- Owner: `AmelieSchreiber`
- Theme: `pink`
- Description: `A retraining of a TokenGT, embedding space GFlowNets GoT reasoning 4x4M-Soft-MoE model with tropical ring attention and geometric algebra constraints`

Collection items:

- Dataset: <https://huggingface.co/datasets/AmelieSchreiber/toricgt-curated-splits>
- Weights: <https://huggingface.co/AmelieSchreiber/toricgt-checkpoints>

## Download Curated Splits

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 \
conda run --no-capture-output -n tokengt hf download \
  AmelieSchreiber/toricgt-curated-splits \
  --repo-type dataset \
  --local-dir data/curated \
  --max-workers 8 \
  --include "README.md" \
  --include "train.parquet" \
  --include "validation.parquet" \
  --include "test.parquet" \
  --include "manifest.json" \
  --include "split_report.json" \
  --include "split_report.md" \
  --include "niqqud_report.json"
```

## Download Checkpoints

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 \
conda run --no-capture-output -n tokengt hf download \
  AmelieSchreiber/toricgt-checkpoints \
  --repo-type model \
  --local-dir checkpoints/hf/toricgt-checkpoints \
  --max-workers 8
```

## Notes

The dataset repo and checkpoint repo are public collection items. The current
checkpoint file is `toricgt_step_00001000.pt`; later training checkpoints should
be uploaded to the same model repo as they become available.
