# Combinatorial Commutative-Algebra and Topology Metrics

Status: active implementation plan for BPB-first Seq4096 training and later
reasoning phases.

## Purpose

ToricGT should not use toric geometry as a decorative label. The train-time
metrics must expose finite algebraic and topological structure that can be
computed, bounded, ablated, and tied back to BPB. The current implementation
therefore treats a sampled local hidden-state window as a finite combinatorial
object:

1. A chart fan, approximated by a small cyclic chamber system.
2. A monomial exponent table and a finite toric ideal shadow.
3. A Stanley-Reisner complex over chamber coactivation.
4. A multiparameter persistence/Koszul module over local chart actions.
5. A directed Vietoris-Rips flag filtration over reasoning trajectory states.

The training objective remains primarily next-token cross entropy and byte-level
BPB. These metrics are auxiliary controls: they should improve the geometry of
hidden reasoning without dominating the language-model objective or violating
the 16MB Parameter-Golf artifact constraint.

## Mathematical Objects

Let a local hidden-state window be

```text
H = (h_1, ..., h_n),    h_i in R^d.
```

The implementation samples at most `max_points` vectors from a training
sequence, normalizes them, and computes all algebraic/topological quantities on
that small finite set.

### Exact Finite Certificate

For each sampled window, the intended exact object is a finite certificate

```text
C(H) = (A, Delta, R, K_*, d_*, T_1, ..., T_r, F_1 subset ... subset F_L).
```

The pieces have the following meanings:

1. `A` is a finite exponent configuration, hence a semigroup algebra shadow
   `k[N A]`.
2. `Delta` is a finite simplicial complex, usually the cyclic fan complex on
   chamber labels.
3. `R` is a small set of verified toric binomial relations `x^u - x^v` with
   `A u = A v`.
4. `(K_*, d_*)` is a finite chain complex over a field, used for exact
   sidecar homology audits.
5. `T_i` are local endomorphisms of the sampled module, used to build Koszul
   differentials.
6. `F_1 subset ... subset F_L` is a finite filtered graph or flag complex.

The certificate is algebraically valid when:

```text
A u = A v for every binomial x^u - x^v in R,
d_{q-1} d_q = 0 for every q,
T_i T_j = T_j T_i for all i,j when a Koszul exactness claim is made,
F_l subset F_{l+1} for every filtration level l.
```

The training code evaluates differentiable polynomial and rank expressions
attached to these exact finite objects. This is the main rigor guardrail: a
loss is not considered a mathematical loss unless its finite certificate and
its differentiable evaluation are both defined and logged.

### Rigorous CCA and Combinatorial Topology Background

The finite commutative-algebra object behind the train-time metric is a
semigroup algebra. Let

```text
A = [a_1 ... a_m] in Z^{s x m}
```

be the deterministic exponent matrix assigned to the current chamber system.
It defines the monomial map

```text
phi_A : k[x_1, ..., x_m] -> k[t_1^{+-1}, ..., t_s^{+-1}],
phi_A(x_i) = t^{a_i}.
```

The toric ideal is the kernel

```text
I_A = ker(phi_A)
    = < x^u - x^v : A u = A v, u,v in N^m >.
```

Thus a binomial relation is not a metaphor: it is an equality in the affine
semigroup algebra `k[N A]`. The training loop cannot afford a Groebner basis on
every step, so it uses a fixed low-degree relation set

```text
R = {(i,j,k,l) : a_i + a_j = a_k + a_l}
```

or a nearest exact sidecar-certified approximation. For chamber logits
`z_i(h)`, the differentiable toric-ideal residual is

```text
L_binom(H) =
  mean_{(i,j,k,l) in R} (z_i + z_j - z_k - z_l)^2
  / (mean_i z_i^2 + eps).
```

Small `L_binom` means the hidden chart respects the additive relations of the
finite exponent configuration. In sidecar audits, the relation is checked
exactly by verifying `A u = A v`.

The combinatorial topology object is a finite simplicial complex
`Delta` on chamber vertices `[m]`. Its Stanley-Reisner ideal is

```text
I_Delta = < x_F = prod_{i in F} x_i : F notin Delta >,
```

and the face ring is

```text
k[Delta] = k[x_1, ..., x_m] / I_Delta.
```

The trainer constructs a cyclic fan complex as the minimal stable early finite
complex: adjacent chambers and diagonal self-pairs are faces, while
nonadjacent chamber pairs are nonfaces. If `p_i(h_t)` is the soft chamber mass
at hidden point `h_t`, then

```text
C_ij(H) = (1/n) sum_t p_i(h_t) p_j(h_t)
```

is a soft coactivation matrix. The Stanley-Reisner loss is

```text
L_SR(H) = mean_{ {i,j} notin Delta } C_ij(H).
```

This is a direct differentiable relaxation of the condition that nonfaces
should have zero monomial mass in `k[Delta]`.

For exact topology, form chain groups over a field, currently `F_2` in sidecar
audits:

```text
C_q(Delta; F_2) = span_F2 { oriented q-simplices of Delta }.
```

The boundary map is

```text
partial_q [v_0, ..., v_q]
  = sum_{r=0}^q (-1)^r [v_0, ..., omit(v_r), ..., v_q].
```

A valid chain complex satisfies

```text
partial_{q-1} partial_q = 0.
```

Homology is

```text
H_q(Delta) = ker(partial_q) / im(partial_{q+1}),
beta_q = dim H_q.
```

The training loop uses cheap graph/Laplacian proxies for `beta_0`, `beta_1`,
and Euler characteristic

```text
chi = beta_0 - beta_1 + beta_2 - ...
    = f_0 - f_1 + f_2 - ...
```

while periodic analyses compute exact finite-field ranks on small sampled
complexes. The loss component

```text
L_topology(H)
```

penalizes violations of nested directed filtrations, unstable directed edges,
and soft triangle inconsistencies. It is allowed to influence training only
because the exact chain-complex audit can verify or falsify the differentiable
evaluation.

The Koszul part attaches local endomorphisms `T_1, ..., T_r` to the sampled
module spanned by hidden states. For commuting parameters, the Koszul complex
has differentials

```text
d_q : M tensor wedge^q(k^r) -> M tensor wedge^{q-1}(k^r),
d(m tensor e_{i_1} wedge ... wedge e_{i_q})
  = sum_l (-1)^{l-1} T_{i_l} m tensor e_{i_1} wedge ... omit(e_{i_l}) ... wedge e_{i_q}.
```

If the `T_i` commute, then `d_{q-1} d_q = 0`. The train-time residual is

```text
L_Koszul =
  ||d_1 d_2||_F^2
  + 1[r >= 3] ||d_2 d_3||_F^2
  + alpha mean_{i<j} ||T_i T_j - T_j T_i||_F^2
  + beta L_rank.
```

`L_rank` is a detached or weakly weighted Buchsbaum-Eisenbud/Fitting-style rank
diagnostic in the main loop, with exact small-matrix audits in the sidecar. This
separates algebraic falsifiability from noisy high-order gradients.

The combined CCA/topology loss is therefore a finite-certificate-backed
differentiable scalar:

```text
L_CCA_top =
    w_binom L_binom
  + w_SR L_SR
  + w_entropy (1 - H(chamber_mass))
  + w_balance ||chamber_mass - uniform||_2^2
  + w_euler |chi_proxy| / m
  + w_K L_Koszul
  + w_top L_topology.
```

All terms are clipped, finite-cleaned, and logged. No term is accepted as
mathematical evidence unless its exact finite certificate is available in the
periodic analysis output and its W&B train metric is visible.

### Full Symbolic Multigraded Resolution Layer

The symbolic layer now required for the training program is the exact
multigraded Stanley-Reisner resolution of the finite chamber complex. Let

```text
S = k[x_1, ..., x_m]
```

with the standard `Z^m` grading `deg(x_i)=e_i`. The chamber complex `Delta`
defines the Stanley-Reisner ideal

```text
I_Delta = < x_F : F notin Delta >
```

and the face ring `k[Delta]=S/I_Delta`. The minimal multigraded Betti numbers
of `k[Delta]` are computed exactly by Hochster's formula:

```text
beta_{i,sigma}(k[Delta])
  = dim_k H~_{|sigma|-i-1}(Delta_sigma; k),
```

where `Delta_sigma` is the induced subcomplex on the vertex subset `sigma`.
The implementation computes these homology ranks over `F_2`, so the
multigraded Betti table, projective dimension, regularity, Betti entropy, and
top multidegree cardinality are exact finite symbolic invariants for the
chosen chamber complex.

The same ideal also has the Taylor free resolution. If

```text
I_Delta = <m_1, ..., m_g>,
```

then the Taylor module in homological degree `p` has one basis element for each
`p`-subset of generators, with multidegree

```text
lcm(m_{i_1}, ..., m_{i_p}).
```

Its differential is

```text
d(e_J) =
  sum_{r=1}^{p} (-1)^{r-1}
    lcm(m_j : j in J) / lcm(m_j : j in J\{j_r}) e_{J\{j_r}}.
```

This is a full symbolic multigraded free resolution, generally nonminimal. The
minimal Hochster table and the Taylor ranks together give:

```text
symbolic_resolution_minimal_total_betti
symbolic_resolution_projective_dimension
symbolic_resolution_regularity
symbolic_resolution_taylor_total_rank_log2
symbolic_resolution_taylor_boundary_terms_log2
symbolic_resolution_nonminimality_log2
```

These exact invariants are discrete certificates. The differentiable part comes
from evaluating the same multigraded polynomial ring on soft chamber variables.
Let

```text
x_i(H) = mean_t p_i(h_t)
```

where `p_i(h_t)` is the soft chamber assignment. For a squarefree monomial
`x_F`, define

```text
x_F(H) = prod_{i in F} x_i(H).
```

Then the train-time differentiable losses are polynomial maps:

```text
L_SR_poly      = mean_{F minimal nonface} x_F(H)
L_Taylor_lcm   = mean_{J={a,b}} x_{lcm(m_a,m_b)}(H)
L_HilbertBetti = c(beta, pdim, reg) * (L_SR_poly + 0.35 L_Taylor_lcm).
```

`c(beta, pdim, reg)` is a constant derived from the exact symbolic resolution.
Thus the symbolic resolution is used twice: exactly for audits and
differentiably as coefficients/targets for polynomial losses over soft
multigraded coordinates.

### DG-Algebra Structure

The Taylor resolution is not merely a graded list of free modules; it is a
differential graded algebra. Let `F_Tay` be the Taylor resolution of
`S/I_Delta`. It has a homological grading, a multigrading, a differential

```text
d : F_p -> F_{p-1},
```

and a product

```text
mu : F_p tensor F_q -> F_{p+q}.
```

For basis elements `e_A` and `e_B` indexed by generator subsets, the product is
zero when repeated generators would occur and otherwise has multidegree

```text
deg(e_A e_B) = lcm(m_i : i in A union B).
```

The DG-algebra identities are exact:

```text
d^2 = 0,
d(ab) = d(a)b + (-1)^{|a|} a d(b),
ab = (-1)^{|a||b|} ba.
```

The periodic analysis reports exact residuals for these identities:

```text
toric_cca_symbolic_dg_d_squared_residual = 0
toric_cca_symbolic_dg_leibniz_residual = 0
```

because they are true symbolic identities of the Taylor DG algebra. The
train-time differentiable part is the DG augmentation/ideal mass. Let

```text
epsilon : F_Tay -> k[Delta]
```

be the augmentation. Basis elements in positive homological degree belong to
the augmentation ideal. Evaluated on soft chamber variables, the differentiable
DG pressure is

```text
L_DG_aug(H)
  = mean_{e_A in F_Tay, |A|>0} x_{deg(e_A)}(H),
```

implemented in compressed form by exact Taylor multidegree counts. This is not
sampled: every Taylor basis multidegree contributes through its exact
multiplicity, with equal lcm multidegrees aggregated before polynomial
evaluation.

### Why These Losses Can Improve BPB

The BPB objective is empirical cross entropy per UTF-8 byte:

```text
BPB(theta) = (1 / log 2) E[-log p_theta(b_target | prefix, order)] / bytes.
```

Auxiliary algebraic losses can help only if they reduce the conditional entropy
of the next byte under the same score-before-update information. The intended
mechanism is representation regularization, not adding hidden evaluation
information. Let `Z_theta` be hidden chart variables produced from the legal
prefix. A useful auxiliary loss should increase the predictive stability of
those variables while decreasing irrelevant chart noise:

```text
H(B_next | prefix)
  >= H(B_next | Z_theta(prefix))
  >= H(B_next | Z_clean(prefix)).
```

The CCA/DG losses target the gap between `Z_theta` and `Z_clean`:

1. Stanley-Reisner generator losses suppress forbidden chamber coactivations:

```text
L_SR_poly = mean_{F minimal nonface} prod_{i in F} x_i(H).
```

This makes active byte features less entangled, so fewer incompatible
predictors compete inside the tied output embedding.

2. Toric binomial losses enforce exact semigroup relations:

```text
L_binom = mean_{Au=Av} (z^u - z^v)^2.
```

In log coordinates this is the implemented additive relation. Equivalent paths
to the same multigrade should agree, reducing variance across equivalent
contexts.

3. Taylor/DG augmentation mass penalizes high-degree forbidden products:

```text
L_DG_aug = mean_{e_A in F_Tay, |A|>0} x_{deg(e_A)}(H).
```

This discourages representing a byte decision as a large inconsistent
conjunction of chamber features. Compact conjunctions are easier for the
small recurrent model and tied output embedding to decode.

4. Differentiable Koszul/Fitting/Buchsbaum-Eisenbud rank losses encourage local
parameters to behave like a regular sequence:

```text
L_K =
  ||d_1 d_2||_F^2
  + 0.35 mean_{i<j} ||T_i T_j - T_j T_i||_F^2
  + 0.10 L_Fitting
  + 0.10 L_BE-rank
  + 0.04 L_BE-mult
  + 0.02 L_Betti.
```

If local parameters are closer to a regular sequence, then reasoning features
factor more cleanly across position, byte class, phase, and graph-order
coordinates.

The practical BPB test remains strict:

```text
Delta_BPB = BPB_with_aux - BPB_base.
```

`Delta_BPB` must be negative or neutral under matched data, tokenizer,
random-order policy, artifact budget, and validation procedure. Therefore all
DG/CCA losses are micro-weighted from step 0, CE-capped, and increased only
when their W&B slopes correlate with improving OAI validation BPB and no
nonfinite or train-BPB spike appears.

### Finite BGG and Category O Shadow

The Toric BGG part is also finite in the training loop. The intended
representation-theoretic object is not an infinite category inside the model;
it is a small highest-weight skeleton that can be written as matrices.

Let `(Lambda, <=)` be a finite poset. A highest-weight proxy over a field `k`
has:

```text
standard objects Delta(lambda),
costandard objects Nabla(lambda),
simple objects L(lambda),
projective covers P(lambda).
```

The BGG reciprocity pattern is

```text
[P(lambda) : Delta(mu)] = [Delta(mu) : L(lambda)].
```

For train-time use this is represented by an incidence-algebra shadow. The
incidence algebra `I(Lambda; k)` has basis elements `e_{lambda,mu}` for
`lambda <= mu` and multiplication

```text
e_{lambda,mu} e_{nu,rho}
  = 1[mu = nu] e_{lambda,rho}.
```

The standard-filtration metric does not ask the model to classify modules.
Instead it asks a hidden trajectory to respect the partial order:

```text
L_standard_leak
  = mean attention/probe mass from a lambda-state into forbidden mu
    with mu not >= lambda under the certificate poset.
```

The BGG differential shadow comes from a multigraded polynomial ring

```text
S = k[x_1, ..., x_n],      E = exterior(e_1, ..., e_n),
```

and a graded module `M`. On `M tensor E`, the BGG/Koszul-style differential is

```text
d(m tensor eta) = sum_i x_i m tensor contraction(e_i, eta).
```

The exact identity `d^2 = 0` follows from commutativity of the `x_i` and
anticommutativity of exterior contraction. The train-time residual is therefore

```text
L_d2 = || d_hat_{q-1} d_hat_q ||_F^2,
```

where `d_hat_q` is a low-rank hidden-state probe predicting the certificate
differential. This is the smallest robust BGG loss because it is local,
matrix-valued, and falsifiable on toy complexes.

Hypertoric category `O` and oriented-matroid category `O` supply the toric
choice of poset. A hyperplane arrangement gives sign vectors or chambers; the
Gale-dual arrangement gives the Koszul-dual curriculum. The finite metric is:

```text
signature(tau)
  = (Betti histogram, Ext-degree histogram, chamber path,
     standard leakage, Gale-dual label, moment/toric phase summary).
```

Analogical and memory retrieval training then compare signatures by
chain-complex structure rather than only by cosine distance:

```text
L_chain_map = || F_{q-1} d_q - d'_q F_q ||_F^2
L_gale      = distance(signature_A(tau), signature_Gale(A)(tau'))
L_memory    = contrastive_loss(memory_key(tau), memory_key(tau_isomorphic))
```

The useful hierarchy is:

1. `d^2` residual and standard leakage: safe early probes, tiny weights.
2. Betti/Ext signature distillation: audit-backed after exact ranks stabilize.
3. Gale-dual consistency and Euler-Koszul exactness: late phase only.
4. Retrieval/memory use of BGG signatures: post-BPB-gate reasoning phase.

This keeps category `O` mathematically meaningful without pretending that the
deployed micro-LM serializes a category. The final artifact stores only the
base model and byte-accounted modules that improve BPB; BGG/category-O probes
are training-time certificates unless an ablation proves otherwise.

### Toric Monomial Data

Choose a deterministic exponent table

```text
A = {a_1, ..., a_m} subset Z^r or R^r,
```

where `m = num_chambers` is small, usually 6 to 8 inside training. The exponent
table represents a compact Newton-polytope shadow. It is not learned and is not
stored in the exported model.

The associated monomial map is

```text
phi_A : k[x_1, ..., x_m] -> k[t_1^{+-1}, ..., t_r^{+-1}],
phi_A(x_i) = t^{a_i}.
```

The toric ideal is

```text
I_A = ker(phi_A)
    = < x^u - x^v : A u = A v, u,v in N^m >.
```

In training we use short degree-two binomial shadows:

```text
a_i + a_j approx a_k + a_l.
```

For chamber logits `z_i(h)`, the differentiable residual is

```text
L_binom =
  mean_{(i,j,k,l) in R} (z_i + z_j - z_k - z_l)^2
  / mean_i z_i^2.
```

This is a combinatorial commutative-algebra metric: it asks the hidden chart to
respect low-degree binomial relations of the finite toric monomial
configuration. It is intentionally a shadow, not a symbolic Groebner-basis
computation in the training loop.

The exact sidecar version can be checked over a small integer or finite-field
table:

```text
audit_binomial(A, u, v):
    return (A @ u == A @ v)
```

When a larger audit budget is available, a small Markov-basis slice can be
computed by enumerating low-degree `u,v` and reducing duplicate fibers
`A u`. The train-time relation set is then the stable low-degree subset that
appears across adjacent windows or matched seeds.

### Stanley-Reisner Complex

Let `Delta` be the cyclic fan complex on chamber labels. Vertices are chambers,
edges connect adjacent fan rays, and nonfaces are chamber pairs that should not
coactivate in the same local toric chart.

The Stanley-Reisner ideal is

```text
I_Delta = < x_F = prod_{i in F} x_i : F notin Delta >.
```

Given soft chamber probabilities

```text
p_i(h) = softmax(z(h) / tau)_i,
```

define coactivation

```text
C_ij = (1/n) sum_t p_i(h_t) p_j(h_t).
```

The training residual is

```text
L_SR = average_{(i,j) notin Delta} C_ij.
```

This is the differentiable Stanley-Reisner nonface mass. It discourages
collapsed hidden charts where mutually incompatible toric chambers fire
together. The companion diagnostics are chamber entropy, chamber coverage,
allowed edge mass, an Euler-characteristic proxy, and Betti proxies from the
thresholded chamber graph.

The exact commutative-algebra object is the face ring

```text
k[Delta] = k[x_1, ..., x_m] / I_Delta.
```

Its multigraded Betti numbers are governed by Hochster's formula:

```text
beta_{i,sigma}(k[Delta])
  = dim_k H_tilde^{|sigma|-i-1}(Delta_sigma; k),
```

where `Delta_sigma` is the restriction of `Delta` to the vertex set `sigma`.
In training, the code uses one-skeleton Betti proxies because they are cheap
and differentiability-safe. In periodic analysis, exact `F_2` boundary ranks
should be computed from the same thresholded complexes:

```text
beta_q = dim ker partial_q - rank image partial_{q+1}.
```

This makes the Stanley-Reisner metric falsifiable: the soft nonface mass should
move in the same direction as exact forbidden-face counts and Betti instability.

### Koszul and Buchsbaum-Eisenbud Shadows

The module `src/toricgt/koszul_persistence.py` builds local row-stochastic
operators

```text
T_1, T_2, T_3 : M -> M
```

from time shift, metric-neighborhood flow, and toric phase/chart flow. These
operators are a finite proxy for commuting parameters in a multiparameter
persistence module or affine-toric module.

For two operators the Koszul differential is

```text
d_1 = [T_1  T_2],
d_2 = [-T_2 ; T_1].
```

For three operators, the usual Koszul signs give

```text
d_1 = [T_1 T_2 T_3],
d_2 =
  [ -T_2  -T_3   0
     T_1    0   -T_3
      0    T_1   T_2 ],
d_3 = [T_3 ; -T_2 ; T_1].
```

The core exactness residual is

```text
L_exact = ||d_1 d_2||_F^2 + ||d_2 d_3||_F^2.
```

The syzygy/commutativity residual is

```text
L_comm = average_{i<j} ||T_i T_j - T_j T_i||_F^2.
```

Rank diagnostics approximate Buchsbaum-Eisenbud conditions and Fitting-ideal
behavior. These rank terms are detached diagnostics during BPB-first training so
SVDs cannot become the main gradient source. The differentiable path is
primarily exactness plus commutator/syzygy residuals.

The exact algebraic reference is the Koszul complex `K(T_1, ..., T_r; M)`.
If the `T_i` commute, the Koszul differential squares to zero. Its homology
measures failure of the operators to form a regular sequence on the sampled
module:

```text
H_q(K(T; M)) = ker d_q / im d_{q+1}.
```

For a finite free complex

```text
F_s -> ... -> F_1 -> F_0,
```

Buchsbaum-Eisenbud exactness supplies the audit discipline: the expected ranks
must add correctly, and the determinantal/Fitting ideals of the differentials
must have the expected grade. The training loop should not attempt symbolic
determinantal algebra. Instead:

1. Sidecar audits compute exact ranks over `F_2` or rationals on tiny windows.
2. Differentiable surrogates penalize `||d d||_F^2`, commutators, and soft rank
   mismatches.
3. A Koszul or Buchsbaum-Eisenbud loss may gain weight only when the exact
   audit and soft surrogate agree over recent windows.

### Directed Persistent Topology

The module `src/toricgt/topological_reasoning.py` treats a hidden window as a
point cloud with time direction. A radius sweep gives nested Vietoris-Rips/flag
complexes:

```text
K_1 subset K_2 subset ... subset K_L.
```

At radius `rho_l`, soft undirected and directed adjacencies are

```text
S_l(i,j) = sigmoid((rho_l - ||h_i-h_j||) / tau),
D_l(i,j) = sigmoid((rho_l - ||h_i-h_j|| + skew_ij + time_ij) / tau).
```

The loss uses differentiable proxies for:

1. Filtration inclusion, penalizing edges that disappear across the sweep.
2. Flag/simplex closure, penalizing two-step paths not supported by edges.
3. Boundary residuals for oriented 2-simplices.
4. Directed transitivity and directed cycle flux.
5. DEC-inspired conservation: divergence, vorticity drift, kinetic-energy
   drift, Hodge balance, and wedge/interior-product consistency.
6. HDBSCAN-style mutual-reachability stability.
7. Analogical transport between consecutive local complexes.

This is a combinatorial-topology metric because the underlying hard object is a
finite filtered simplicial complex and a directed graph-of-reasoning trajectory.

The exact sidecar construction uses boundary matrices over `F_2` or signed
integer incidence matrices:

```text
partial_1: C_1(K_l) -> C_0(K_l),
partial_2: C_2(K_l) -> C_1(K_l),
partial_1 partial_2 = 0.
```

Persistent inclusions are maps `i_l: K_l -> K_{l+1}` inducing homology maps
`H_q(i_l)`. The training surrogate is useful only if it predicts the same
qualitative events as the exact filtration: component mergers, short-lived
cycle creation, directed cycle flux, and simplex closure failures. This is why
the periodic analysis must render both soft trajectory plots and exact
finite-complex summaries.

### Loss Validity Gates

Every advanced algebra/topology signal has three possible statuses:

1. `metric_only`: logged for visibility; cannot affect gradients.
2. `audit_backed`: exact sidecar audit exists and agrees with the soft metric.
3. `loss_enabled`: allowed into the auxiliary loss after a BPB-safe ablation.

The promotion rule is:

```text
if exact_audit_passes
   and soft_metric_tracks_audit
   and BPB_ablation_delta >= -tolerance:
       allow small loss weight
else:
       keep metric_only or audit_backed
```

This prevents beautiful but unhelpful algebra from steering the BPB-first phase.
For Parameter Golf, a metric is valuable only if it helps reduce OAI validation
BPB, improves an explicitly tracked reasoning slice after the BPB gate, or
diagnoses a failure that leads to a better configuration.

### Exact Sidecar Algorithms

The periodic analysis suite should compute exact integer or finite-field
versions of the same objects used by the differentiable loss. The intended
small-window algorithms are:

```text
low_degree_toric_relations(A, degree_bound):
    fibers = group monomial exponent vectors u by A @ u
    for each fiber with at least two elements:
        emit binomials u - v from a stable spanning tree of the fiber
```

```text
stanley_reisner_audit(coactivation, fan_edges, threshold):
    hard_edges = {(i,j): coactivation[i,j] >= threshold}
    forbidden = hard_edges - fan_edges
    K = flag_complex(hard_edges)
    boundary_1, boundary_2 = incidence_matrices_F2(K)
    beta0 = n_vertices - rank(boundary_1)
    beta1 = nullity(boundary_1) - rank(boundary_2)
    return forbidden_fraction, beta0, beta1, euler(K)
```

```text
koszul_audit(T_1, ..., T_r):
    require ||T_i T_j - T_j T_i|| small for all i < j
    build signed Koszul differentials d_q
    exactness_residual = sum_q rank_or_norm(d_{q-1} @ d_q)
    homology_rank_q = nullity(d_q) - rank(d_{q+1})
```

```text
directed_persistence_audit(points, radii):
    for rho in radii:
        build directed graph and undirected flag complex K_rho
        compute F2 boundary ranks, component mergers, cycle births
        compute directed flux, transitivity, and simplex closure failures
    verify K_rho subset K_rho_next and summarize barcode-like lifetimes
```

The exact values are not gradients. They decide whether a soft metric is
trustworthy. A soft loss can be promoted only when:

```text
corr(soft_metric, exact_audit) >= corr_min
and exact_nonfinite_count == 0
and validation_bpb_delta is not worse than tolerance
and artifact_bytes_delta == 0 for training-only probes.
```

For W&B, the exact audit names should be grouped separately from the training
losses:

```text
advanced/toric_cca_*          # differentiable train-time metrics
analysis_control/exact_cca/*  # sidecar finite algebra/topology audits
```

When these disagree, the exact audit wins: reduce or hold the soft loss weight
and use the plots to diagnose whether the chart basis, threshold, relation set,
or topology radius is wrong.

### Implementation Map

The current bridge maps theory to code as follows:

```text
toric ideal residual
  -> src/toricgt/combinatorial_toric_metrics.py: toric_cca_binomial_residual

Stanley-Reisner nonface mass
  -> toric_cca_stanley_reisner_nonface_mass

face-ring coarse topology
  -> toric_cca_euler_characteristic_proxy
  -> toric_cca_betti0_proxy
  -> toric_cca_betti1_proxy

Koszul-persistence module
  -> src/toricgt/koszul_persistence.py
  -> toric_cca_koszul_loss

directed filtered flag complex
  -> src/toricgt/topological_reasoning.py
  -> toric_cca_topology_loss_component

combined bounded bridge
  -> toric_cca_topology_loss
```

All chart directions and exponent tables are deterministic functions of hidden
states and configuration. No CCA/topology parameters are serialized into the
competition artifact.

## Combined Train-Time Loss

The new bridge in `src/toricgt/combinatorial_toric_metrics.py` computes

```text
L_CCA_top =
    w_binom L_binom
  + w_SR L_SR
  + w_entropy (1 - H(chamber_mass))
  + w_balance L_balance
  + w_euler |chi_proxy| / m
  + w_koszul L_koszul
  + w_top L_topology.
```

The bridge returns the differentiable scalar
`toric_cca_topology_loss` and detached metrics:

```text
toric_cca_binomial_residual
toric_cca_stanley_reisner_nonface_mass
toric_cca_chart_entropy
toric_cca_chamber_coverage
toric_cca_fan_balance_loss
toric_cca_euler_characteristic_proxy
toric_cca_betti0_proxy
toric_cca_betti1_proxy
toric_cca_allowed_edge_mass
toric_cca_koszul_loss
toric_cca_topology_loss_component
```

The Seq4096 Parameter-Golf trainer receives this loss only through the existing
advanced auxiliary-loss channel. That channel has:

```text
ADVANCED_LOSS_SCALE
ADVANCED_LOSS_START_STEP
ADVANCED_LOSS_EVERY
ADVANCED_LOSS_WARMUP_STEPS
ADVANCED_LOSS_MIN_BEST_VAL_BPB
ADVANCED_LOSS_MAX_CE_RATIO
GRAD_CLIP_NORM
```

so the algebra/topology term is bounded relative to the current cross entropy.

Step-0 utilization rule: if CCA/topology is requested for a scratch run, it
must not be hidden behind a later validation-BPB gate. The corrected BPB-first
launcher therefore sets `ADVANCED_LOSS_START_STEP=0` and
`ADVANCED_LOSS_MIN_BEST_VAL_BPB=0`, while keeping a long warmup and a small
CE-ratio cap. This makes the finite algebraic/topological metrics visible from
the first optimizer steps and gives the model a tiny train-time pressure toward
clean toric charts, Stanley-Reisner consistency, Koszul exactness, and directed
filtration coherence. If stability is in doubt, the appropriate fallback is
`ADVANCED_LOSS_LOG_ONLY=1` for a short audit run, not silently suppressing the
metrics by setting runtime scale to zero.

## Wise Utilization Policy

For the BPB-first phase:

1. Keep cross entropy and OpenAI FineWeb validation BPB as the primary objective.
2. Enable CCA/topology metrics from step 0, but with micro weights and a long
   ramp.
3. Cap total auxiliary loss at a tiny ratio of CE, initially `1e-5` to `2e-5`.
4. Log every component to W&B under `advanced/toric_cca_*`.
5. Treat a rising nonfinite flag, sudden BPB stall, or train-BPB spike as a
   reason to reduce the advanced scale, not as a reason to trust the math more.

Current scratch-run starting point after the compile-safe PolarQuant fix:

```text
ADVANCED_LOSS_SCALE=0.00015
ADVANCED_LOSS_START_STEP=0
ADVANCED_LOSS_EVERY=16
ADVANCED_LOSS_WARMUP_STEPS=20000
ADVANCED_LOSS_MIN_BEST_VAL_BPB=0
ADVANCED_LOSS_MAX_CE_RATIO=0.000005
KOSZUL_BGG_LOSS_WEIGHT=0.0001
TORIC_TROPICAL_LOSS_WEIGHT=0.00025
GRAPHCG_LOSS_WEIGHT=0.0005
SLEPIAN_LOSS_WEIGHT=0.0005
ANALOGY_LOSS_WEIGHT=0.000002
GRAD_CLIP_NORM=0.22
POLARQUANT_TRAIN=1
POLARQUANT_TRAIN_START_STEP=0
POLARQUANT_WARMUP_STEPS=0
```

This means the advanced metrics and losses are active from step 0. The force is
still deliberately tiny: at early steps the long warmup makes the runtime scale
near zero, and `ADVANCED_LOSS_MAX_CE_RATIO` prevents the auxiliary objective
from exceeding `5e-6` of the current CE loss. The W&B evidence that this
contract is actually active is:

```text
advanced/runtime_scale_applied > 0 on scheduled advanced steps
advanced/backprop_enabled = 1 on scheduled advanced steps
advanced/toric_cca_topology_loss
advanced/toric_cca_binomial_residual
advanced/toric_cca_stanley_reisner_nonface_mass
advanced/toric_cca_koszul_loss
advanced/toric_cca_topology_loss_component
advanced/koszul_bgg_loss
advanced/aux_loss
advanced/aux_loss_clamped
advanced/toric_cca_bridge_failed = 0
```

For the post-`<=1.2` reasoning phase:

1. Increase CCA/topology frequency before increasing raw weight.
2. Let the topology component influence graph-of-thought trajectory quality,
   memory retrieval keys, and analogical transport.
3. Add sidecar exact audits: F2 boundary ranks, persistent diagrams, toric ideal
   residual histograms, and Stanley-Reisner nonface heatmaps.
4. Use GraphCG disentanglement to make the chart basis better conditioned
   before asking toric fan metrics to become stronger.
5. Keep a natural-language BPB slice and an OAI competition BPB slice separate
   from reasoning-slice BPB.

## Pseudocode

```python
def combinatorial_toric_loss(hidden, positions):
    windows = sample_windows(hidden, max_windows=2, max_points=16)
    toric_terms = []
    for points in windows:
        directions = deterministic_chart_directions(d_model, num_chambers)
        logits = normalize(points) @ directions.T
        probs = softmax(logits / temperature)

        # Toric ideal shadow: x_i x_j - x_k x_l.
        relations = approximate_degree_two_binomials(exponent_table)
        binom = mean_square(logits[i] + logits[j] - logits[k] - logits[l])

        # Stanley-Reisner shadow for cyclic fan.
        coactivation = probs.T @ probs / len(points)
        nonface = mean(coactivation[pairs_not_in_fan])

        # f-vector and Betti diagnostics.
        chamber_graph = threshold(coactivation)
        euler = vertices - edges + triangles
        betti0, betti1 = graph_betti_proxies(chamber_graph)

        toric_terms.append((binom, nonface, euler, betti0, betti1))

    koszul = koszul_persistence_loss(hidden, positions)
    topology = reasoning_step_topology_loss(hidden)
    return weighted_sum(toric_terms, koszul, topology).clamp(0, max_loss)
```

## Visualization Contract

Periodic analyses should render:

1. A dark-mode Stanley-Reisner heatmap: allowed fan edges vs nonface mass.
2. A toric ideal residual histogram over binomial relations.
3. A symbolic resolution complex panel: cyclic fan/nonface graph, exact
   Hochster Betti table, and full Taylor multidegree-count heatmap.
4. A chamber-entropy and chamber-coverage timeline.
5. A 3D PCA/UMAP reasoning trajectory with directed simplices overlaid.
6. A triangle/tetrahedron panel with correct filled heatmaps for BPB,
   topology, GraphCG coherence, and CCA residuals.
7. A persistence panel with Betti0/Betti1 proxies and exact F2 audits where
   sidecar runtime allows.
8. A W&B summary panel with `openai_parameter_golf/bpb` first, then
   `advanced/toric_cca_*`, then other auxiliary categories.

## Current Gaps and Next Updates

Implemented now:

1. Parameter-free CCA/topology bridge in `src/toricgt/combinatorial_toric_metrics.py`.
2. Finite, differentiable unit test coverage.
3. Seq4096 patcher hook so the live trainer can use the bridge without adding
   parameters to the exported artifact.

Next recommended updates:

1. Add exact F2 homology audits to periodic sidecar analysis for the same
   chamber complexes used in training.
2. Plot Stanley-Reisner heatmaps and binomial residual histograms in
   `evaluate_reasoning_geometry_suite.py`.
3. Add W&B organization entries for `advanced/toric_cca_*` so these metrics sit
   under a single visible advanced algebra/topology group.
4. Add a controller rule: if `toric_cca_topology_loss` rises while BPB improves,
   hold; if it rises with BPB degradation, reduce `ADVANCED_LOSS_SCALE`; if it
   falls while BPB improves, consider increasing frequency before increasing
   weight.
