# Toric Slepian / PSWF Audits

## Purpose

The noncommutative torus projection gives each random-order reasoning path a
finite phase trajectory on a commutative torus shadow.  The Slepian/DPSS audit
asks whether that finite phase path has coherent time-band-limited structure or
whether its phase energy is diffuse.  This is the computationally tractable
finite analogue of using prolate spheroidal wave functions on torus-projected
phase leaves.

## Implemented Objects

- `src/toricgt/slepian_torus.py` builds the discrete time-band limiting kernel
  with diagonal `2W` and off-diagonal
  `sin(2*pi*W*(m-n))/(pi*(m-n))`.
- `toric_slepian_audit` maps a projected phase path to a real low-character
  signal, projects it onto leading DPSS modes, and reports concentration,
  leakage, mode entropy, effective mode count, eigenvalues, coefficients, and
  reconstruction.
- `scripts/evaluate_reasoning_geometry_suite.py` writes
  `*_toric_slepian_audit.png` and W&B summary metrics under
  `analysis/mean_toric_slepian_*`.
- `src/toricgt/music.py` uses the same DPSS envelope for deterministic
  torus-constrained synth dynamics and writes JSON-safe Slepian metadata.

## Training Policy

This remains audit-first.  We do not optimize a Slepian loss in the competition
model until ablations show that increasing phase concentration improves BPB,
answer likelihood, or graph-of-thought robustness.  If those ablations pass,
the loss should be low weight and branch-local:

```text
L_slepian = lambda_sl * (1 - C_sl) + lambda_entropy * max(0, H_sl - H_target)
```

where `C_sl` is DPSS concentration and `H_sl` is mode entropy.  The objective
should be disabled for batches without toric/graph reasoning structure.
