# Homotopy Type Theory Analogue For ToricGT Reasoning

Status: proposal and implementation contract for a later research iteration.  This
document does not claim that the current byte-level Parameter-Golf model
implements HoTT.  It defines the mathematical analogue that should guide the
next topology-aware graph-of-thought training pass.

Primary background source: the Homotopy Type Theory book,
<https://homotopytypetheory.org/book/>.

## Why HoTT Is Relevant

GraphCG argues that analogical reasoning appears when relations in embedding
space align geometrically and Transformer layers act like reusable mappings.
The ToricGT topology extension strengthens this picture: instead of treating an
analogy as one vector equation, it treats a reasoning trajectory as a family of
paths, loops, higher cells, and quotient-like identifications.

HoTT gives a principled language for that family:

- a problem domain behaves like a type;
- a concrete hidden reasoning state behaves like a term;
- an analogy between two states behaves like an identity/path;
- a reusable reasoning strategy behaves like a dependent function;
- equivalence-preserving transfer behaves like a univalence-style principle;
- two different reasoning trajectories that solve the same problem behave like
  higher paths/homotopies between paths;
- graph symmetries, root-template ambiguity, and proof normalization behave
  like quotient or higher inductive constructions.

The model-side analogue is not symbolic proof checking.  It is a measured
geometry of paths and higher paths in embedding space that can be made visible
through persistent directed complexes.

## Analogue Dictionary

| HoTT object | ToricGT analogue | Observable proxy |
| --- | --- | --- |
| Type \(A\) | task family or local graph-of-thought state space | dataset/task node, graph projection prefix |
| Term \(a:A\) | hidden state \(h_t\) or terminal candidate | trajectory vertex |
| Identity type \(a =_A b\) | analogy/path between states | relation arrow \(h_b-h_a\), directed edge |
| Path composition | multi-step reasoning composition | directed 2-step closure and transitive residual |
| Inverse path | reverse reasoning transform | skew/asymmetry under reversed directed edge |
| Higher path | homotopy between reasoning trajectories | persistent loops and filled triangles |
| Equivalence \(A\simeq B\) | reusable transfer between domains | low Dirichlet energy under shared relation class |
| Univalence heuristic | equivalent domains admit transfer | stable simplex signatures across task domains |
| Higher inductive quotient | graph/relabeling/root ambiguity quotient | equivariant canonical graph hash and orbit diagnostics |

## Practical Training Consequences

1. Treat repeated relation classes as path families, not just displacement
   vectors.
2. Track whether path composition closes into directed 2-simplices.
3. Penalize unstable analogical features only after density-stability gating;
   otherwise the model collapses rare but valid solution modes.
4. Prefer homotopy-level equivalence diagnostics over single terminal accuracy
   when evaluating graph-of-thought branches.
5. Preserve noncommutativity: a path \(p:a\to b\) and reverse path \(p^{-1}\)
   need not have identical local computation in a directed reasoning flow.

## Concrete Next-Iteration Objective

For a branch trajectory \(h_0,\ldots,h_T\), define local windows
\[
W_t=\{h_{t-r},\ldots,h_t,\ldots,h_{t+r}\}.
\]
Each \(W_t\) builds a directed filtered flag complex
\[
K_t(\rho_1)\hookrightarrow K_t(\rho_2)\hookrightarrow\cdots\hookrightarrow
K_t(\rho_L).
\]
Two trajectories \(\gamma,\gamma'\) that solve the same problem should have
compatible persistent summaries:
\[
d_{\rm bottleneck}\bigl(\mathrm{PH}(K_\gamma),\mathrm{PH}(K_{\gamma'})\bigr)
\lambda_\Omega\|\Omega_\gamma-\Omega_{\gamma'}\|
\lambda_D|E_D(\gamma)-E_D(\gamma')|.
\]
The present implementation uses differentiable surrogates for this expression:
nearest-neighbor barcode pressure, soft triangle closure, inclusion residuals,
directed chain-map commutators, HDBSCAN stability, and Dirichlet energy.

## HoTT-Inspired Evaluation

The geometry suite should include:

- path-level: 3D reasoning trajectories with terminal solution markers;
- homotopy-level: loops/cycle-rank over radius sweeps;
- quotient-level: equivariance-orbit comparisons and graph hash clusters;
- transfer-level: cross-domain simplex signature similarity;
- univalence heuristic: whether two task domains with matched persistent
  signatures have lower transfer loss than unmatched domains.

## Non-Goals

- Do not add a symbolic HoTT kernel to the Parameter-Golf artifact.
- Do not force every reasoning graph into a simply connected shape.
- Do not make topology losses dominate BPB during early compression training.
- Do not replace exact graph/equivariance tests with visual topology plots.
