# BPB-Focused Advanced Mathematics Training Modifications

This checklist is for the OAI Parameter-Golf baseline adaptation inside ToricGT. The active run should not be interrupted. These changes must be picked up by the next fresh restart at the 1.5K-step gate.

## Implementation Checklist

- [x] Add a BPB-gated Lagrangian auxiliary controller to the ToricGT OAI sidecar.
  - Treat advanced mathematical losses as constrained residuals.
  - Maintain online dual variables per family.
  - Increase pressure only when residuals are above target and the BPB/NLL safety gate permits it.
  - Log dual variables, multipliers, residuals, targets, and safety gates to W&B.

- [x] Add a toric fan curriculum for tropical/toric geometry.
  - Stage 0: active-face CE, margin, entropy, and moment alignment only.
  - Stage 1: add bends, affine wall/Coxeter, and leaf coherence.
  - Stage 2: add binomial/divisor-like relations and braid terms.
  - Expose staged losses so the sidecar can select the correct gradient-bearing objective.

- [x] Replace raw memory CE dominance with sheaf-theoretic retrieval gating.
  - Use selected-memory agreement across GraphCG chart, toric phase, topology, GUDHI vectorized persistence, DAG shape, and derived-category features.
  - Gate CE when structural gluing evidence is weak.
  - Keep distillation and quality calibration active so the retrieval head still learns.
  - Log sheaf gluing score, gate, and selected structural similarities.

- [x] Add an online derived-signature distillation head.
  - Use exact/offline-derived-style invariants available from the bounded derived-category feature summary as detached teacher targets.
  - Train a tiny predictor from hidden summaries.
  - Use it as a low-weight online distillation target, not as direct CAS execution inside the training step.

- [x] Strengthen BPB-preserving graph flattening calibration.
  - Keep graph-in/graph-out first.
  - Keep flattening scoped to OAI FineWeb BPB scoring.
  - Add explicit BPB-safety metadata and adaptive config knobs so the next-run controller can increase graph structure only when flattening lift is nonnegative.

- [x] Update the adaptive campaign controller to enable these changes for the next iteration.
  - Add environment overrides for the controller, fan curriculum, sheaf memory gate, derived signature head, and flattening safety calibration.
  - Keep every metric observable for the periodic report.

- [x] Run compile and smoke tests without interrupting the active tmux training session.

## Expected Training Effect

These changes make the advanced mathematics behave like a shaped constraint system rather than a pile of fixed auxiliary losses. The intended BPB effect is early-likelihood protection with later structural pressure: the base byte model descends first; the toric/tropical/topological/category-theoretic objectives then organize hidden states where evidence says they help prediction, retrieval, and graph decoding.
