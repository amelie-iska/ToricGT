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

The public dataset repo currently contains the dataset card, manifest, split
reports, niqqud audit, and architecture image. The local split Parquet files are
large (`train.parquet` is about 40GB; validation and test are about 5.1GB each),
so upload or resume those files with `hf upload-large-folder` before expecting
the command below to download Parquet data from the Hub.

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

Resume the large Parquet upload:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 \
conda run --no-capture-output -n tokengt hf upload-large-folder \
  AmelieSchreiber/toricgt-curated-splits \
  data/curated \
  --type dataset \
  --no-private \
  --num-workers 8 \
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

The `oai` Parameter Golf run publishes only the best current checkpoint to the
same model repo as `parameter_golf_oai_best.pt`, with manifest
`parameter_golf_oai_best.json`. Promotion is gated by validation BPB plus the
Kolmogorov-style prediction-target NCD metric, so worse interval checkpoints
are not uploaded over the current best.

## Notes

The dataset repo and checkpoint repo are public collection items. The current
graph-research checkpoint file is `toricgt_step_00001000.pt`; the competition
branch's live candidate is `parameter_golf_oai_best.pt`.
