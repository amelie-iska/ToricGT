# Toric Geometry Training Signal

This plan records the `oai` branch implementation of the toric-geometry
training signal.  It is intentionally a low-rank auxiliary layer on top of the
Parameter-Golf adapter: BPB remains primary, while toric geometry supplies
auditable structure for reasoning decisions.

## Constraints

- Keep the deployable Parameter-Golf artifact dense and compact.
- Keep toric geometry probes training-only; they are excluded from packed
  challenge artifacts.
- Resume old checkpoints without remapping existing optimizer slots.
- Do not resume training until the full analysis suite has finished and the
  generated plots have been reviewed.
- The requested "there be dragons" responding-onlooker ablation is satisfied by
  absence: repository search currently finds no matching text, prompt role, or
  observer path.

## Implemented Signal

The training-only module is `src/toricgt/toric_geometry_tasks.py`.

- `LowRankToricGeometryProbe`: a small low-rank probe attached to hidden states.
- Straight-through fake quantization on probe matrices, default 6-bit.
- Deterministic Newton-polytope exponent table.
- Synthetic phase teacher from irrational torus coordinates.
- Newton active-face prediction and active-face margin loss.
- Moment-map alignment loss.
- Cartier-divisor-style bend loss from second differences of moment paths.
- Toric ideal/binomial relation checks from approximate exponent relations.
- Affine Coxeter wall/reflection consistency using simple roots.
- A2 braid path consistency from alternating affine reflections.
- Noncommutative phase-foliation residual from projected torus leaf increments.

The implemented auxiliary loss is

```text
L = L_base
  + lambda_toric * (
      lambda_fan L_fan
    + lambda_bend L_bend
    + lambda_binom L_binom
    + lambda_moment ||mu_hat - mu_star||_2^2
    + lambda_coxeter L_coxeter
    + lambda_braid L_braid
    + lambda_leaf L_leaf
    )
```

`lambda_toric` is scheduled from zero after the early BPB-capture phase:

- steps 1000-3000: off;
- steps 3000-6000: tiny warm start;
- steps 6000-12000: low;
- steps 12000+: conservative active weight.

## Metrics

The trainer logs these to W&B:

- `train/toric_geometry_loss`
- `train/toric_geometry_loss_weight`
- `train/toric_fan_loss`
- `train/toric_active_face_ce`
- `train/toric_active_face_margin`
- `train/toric_active_face_entropy`
- `train/toric_bend_loss`
- `train/toric_bend_magnitude`
- `train/toric_binomial_loss`
- `train/toric_binomial_residual`
- `train/toric_moment_loss`
- `train/toric_coxeter_loss`
- `train/toric_affine_wall_distance`
- `train/toric_braid_loss`
- `train/toric_leaf_residual`
- `train/toric_probe_rank`
- `train/toric_probe_quant_bits`

Interpretation:

- Falling fan CE with nonzero margin means hidden states are learning stable
  exposed Newton faces.
- Moderate active-face entropy is desirable; collapse to one face is bad, while
  maximal entropy means the probe is not organizing decisions.
- Falling bend and binomial losses mean the learned moment path is becoming a
  more coherent piecewise-linear toric shadow.
- Falling Coxeter/braid losses mean wall-reflection actions are becoming
  consistent in the moment chart.
- Falling leaf residual means noncommutative toric memory is moving coherently
  along projected phase leaves instead of behaving as decorative sinusoidal
  features.

## Analysis Suite

`scripts/evaluate_reasoning_geometry_suite.py` now computes empirical toric
shadows even for checkpoints trained before the probe existed:

- occupied fan cells;
- fan-cell entropy;
- fitted active-face margins;
- minimum margin;
- bend magnitudes;
- local slope residuals.

It also renders `*_toric_shadow_audit.png`, containing:

- active Newton fan cells along the reasoning path;
- tropical margins and bend magnitudes;
- branch fan coverage vs. branch BPB;
- fan entropy, bend magnitude, and phase-leaf residual by branch.

These plots are logged to W&B under `analysis/images/*` when the analysis run
uses `--wandb`.

## Checkpoint Resume Policy

Older checkpoints may not contain `toric_geometry_probe.*`.  Training and
analysis loaders treat those keys as allowed missing keys and initialize the
probe from the current config.  The probe is appended after existing deployable
modules so optimizer-state padding adds new states without shifting old ones.

## Next Evaluation Gate

Before training resumes:

1. run pycompile and targeted unit tests;
2. run the geometry analysis suite on the latest checkpoint;
3. inspect all generated plot classes, especially toric shadow, toric phase
   simplicial trajectory, exact persistence morphisms, step-radius hierarchy,
   triangles, and tetrahedra;
4. record the summary in the assistant report;
5. only then restart tmux training from the selected checkpoint.

