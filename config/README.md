# ToricGT Configs

These YAML files provide script defaults. Command-line flags still override the
values loaded from `--config`.

## Training

The configured 30M-class training plan has two executable training YAMLs.
Both use `batch_size: 2` and `grad_accum_steps: 16`, so one optimizer step
consumes 32 graph records.

| Phase | Config | Optimizer steps | Approx. train epochs | GFlowNet TB steps | Unsupervised-only steps |
| --- | --- | ---: | ---: | ---: | ---: |
| Warmup | `train.full_30m_warmup.yaml` | 2,000 | 0.013812 | 2,000 at weight 0.01 | 0 |
| Braided experts | `train.full_30m_cyclic.yaml` | 98,000 | 0.676798 | 98,000 at weight 0.05 | 0 |
| Total | both | 100,000 | 0.690610 | 100,000 | 0 |

The train split has 4,633,582 graph records and about 11.607B estimated source
text tokens. These epoch counts are pass-fraction estimates over the curated
train split, not strict full-dataset epochs. The current implementation does
not have a separate unsupervised-only phase; the base objective is supervised
or self-supervised graph reconstruction from curated graph records, with
GFlowNet trajectory balance as an auxiliary objective when enabled.

Run the warmup:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src WANDB_PROJECT=toricgt \
  python scripts/train.py --config config/train.full_30m_warmup.yaml
```

Then resume into braided expert training:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src WANDB_PROJECT=toricgt \
  python scripts/train.py --config config/train.full_30m_cyclic.yaml
```

To continue from a newer checkpoint, override only `--resume`:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src WANDB_PROJECT=toricgt \
  python scripts/train.py \
  --config config/train.full_30m_cyclic.yaml \
  --resume checkpoints/toricgt_full_30m/toricgt_step_00008000.pt
```

## Inference-Time Scaling

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/test_time_scaling.py --config config/inference.test_time_scaling.yaml
```

Use `--device cpu` for a non-GPU check, or `--synthetic --batches 2 --budgets 1 2`
for a tiny smoke run that does not auto-curate held-out data.
