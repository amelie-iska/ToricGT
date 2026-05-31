# GraphCG, Analogical Reasoning, And Directed Persistent Topology

This plan separates two papers that serve different roles in ToricGT.

- GraphCG (`assets/2401.17123v1.pdf`, local clone `amelie-iska/GraphCG/`)
  is a controllable-generation training paradigm.  It learns a disentangled
  latent basis whose directions are steerable factors or concepts.
- Emergent Analogical Reasoning in Transformers
  (`assets/2602.01992v5.pdf`) is a mechanistic account of analogy.  It argues
  that analogical reasoning emerges when relational structures become aligned
  in embedding space and when Transformer layers apply a functor-like map.
- ToricGT combines them by computing analogical topology in the learned
  GraphCG coordinate chart.  Tropical and toric geometry provide sharp
  polyhedral cells, phase winding, and noncommutative directed flow; persistent
  topology describes the nested relation and reasoning-step complexes living in
  those coordinates.

## Paper Summaries

### GraphCG

GraphCG assumes a pretrained graph deep generative model with latent code
`z`.  It learns direction embeddings `d_i` and an editing function
`h(z,d_i,alpha)` so that moving along `d_i` changes one semantic factor while
leaving unrelated factors comparatively stable.  Its contrastive objective
constructs positives by editing the same code with the same direction and step
size, and negatives by changing the direction and/or step size.  The
implementation additionally decorrelates directions and can impose sparsity.

For ToricGT, we do not serialize a separate graph generator inside the
Parameter-Golf artifact.  Instead, hidden states and relation arrows are the
latent codes.  A small learned basis
\[
  B=[d_1,\ldots,d_r]\in\mathbb R^{d\times r}
\]
defines the competition-safe concept chart.  The GraphCG-inspired losses are:

\[
  \mathcal L_{\rm code}
  =\operatorname{CE}\left(
    \frac{\langle \operatorname{norm}(z+\alpha d_i),
                    \operatorname{norm}(z+2\alpha d_j)\rangle}{T},
    i
  \right),
\]
\[
  \mathcal L_{\rm orth}=\|B^\top B-I\|_F^2,\qquad
  \mathcal L_{\rm cov}=\|\operatorname{Corr}(ZB)-I\|_F^2.
\]

These losses make axes identifiable, approximately orthogonal, and useful as a
low-dimensional coordinate system for later topology.

### Emergent Analogical Reasoning

The analogical-reasoning paper formalizes analogies as mappings between
relational structures.  In its synthetic task, entities are partitioned into
categories, relations connect entities inside a category, and a functor-like
token maps one category to another.  The paper identifies two mechanisms:

1. structural alignment in embedding space, measured by lower Dirichlet energy
   over the functor-induced adjacency;
2. functor application, where attention retrieves the source entity and the
   residual stream approximately adds the functor vector:
   \[
     e_{\rm target}\approx e_{\rm source}+f.
   \]

For ToricGT, the key lessons are optimization-sensitive: analogical reasoning
often appears later than ordinary fitting, benefits from moderate regularity,
and can be diagnosed before end-task success by decreasing Dirichlet energy.
We therefore treat analogy losses as phased auxiliary signals, not as the
primary BPB objective.

## ToricGT Synthesis

Let `h_t` be a prefix-visible hidden state in a random-order autoregressive
trajectory.  The GraphCG basis gives chart coordinates
\[
  \xi_t=B^\top h_t,\qquad
  \alpha_t=B^\top\frac{h_{t+s}-h_t}{\|h_{t+s}-h_t\|+\epsilon}.
\]
All scale-sensitive topology is computed on `xi_t` or `alpha_t`, not on raw
hidden states.  This makes the topology a structure over concepts rather than
over arbitrary coordinates.

Tropical attention partitions hidden space into polyhedral cells:
\[
  y(h)=\max_r\{\langle a_r,h\rangle+b_r\}.
\]
The active index is an exposed face of a Newton polytope, and the top-two
margin is distance-to-wall information for that tropical cell.  In the GraphCG
chart, these cells segment concept space into sharp reasoning chambers.

The finite noncommutative-toric memory adds irrational phase probes
\[
  u_s=e^{2\pi i\theta s},\qquad v_s=e^{2\pi i\beta s},
\]
with a quadratic cocycle surrogate.  For rationally independent
`1,theta,beta`, the phase orbit is equidistributed on the torus.  Projecting
reasoning steps to
\[
  ((R+r\cos 2\pi \beta s)\cos 2\pi\theta s,\,
   (R+r\cos 2\pi \beta s)\sin 2\pi\theta s,\,
   r\sin 2\pi\beta s)
\]
shows pseudo-periodic coverage and recurrence of reasoning positions.  The
directed topology uses an antisymmetric toric form so that reversing a
reasoning edge changes the complex.

## Directed Persistent Complexes

For each reasoning window
\[
  W_s=\{\xi_s,\ldots,\xi_{s+w-1}\},
\]
and radius levels \(\rho_1<\cdots<\rho_L\), ToricGT builds a nested
Vietoris--Rips/flag hierarchy:
\[
  K_s(\rho_1)\subseteq\cdots\subseteq K_s(\rho_L).
\]
The symmetric edge score is
\[
  A_\ell(i,j)=\sigma((\rho_\ell-d_{ij})/\tau)(1-\delta_{ij}),
\]
where distances are median-normalized.  Directionality comes from time order
and a noncommutative skew form:
\[
  A_\ell^\rightarrow(i,j)=
  \sigma((\rho_\ell-d_{ij}+\lambda\Omega_{ij}+\eta\,\operatorname{sign}(j-i))/\tau)(1-\delta_{ij}).
\]

The differentiable training terms are:

- inclusion residual: \(\|\max(0,A_\ell-A_{\ell+1})\|^2\);
- soft triangle closure and boundary residual;
- Dirichlet energy: \(\sum A_{ij}d_{ij}^2/\sum A_{ij}\);
- directed transitive residual and directed cycle flux;
- radius-HDBSCAN mutual-reachability stability gate.

## Analogical Maps Between Complexes

For consecutive windows \(K_s\) and \(K_{s+1}\), the implementation builds a
soft transport map
\[
  P_{ij}
  =
  \frac{\exp(-\|\xi_i^{(s)}-\xi_j^{(s+1)}\|/\tau)}
       {\sum_k\exp(-\|\xi_i^{(s)}-\xi_k^{(s+1)}\|/\tau)}.
\]
It then pushes source adjacencies forward:
\[
  \widehat A_{\ell}^{(s+1)}=P^\top A_\ell^{(s)}P,\qquad
  \widehat A_{\ell}^{\rightarrow,(s+1)}
  =P^\top A_\ell^{\rightarrow,(s)}P,
\]
and penalizes residuals against the target complex:
\[
  \mathcal L_{\rm map}
  =
  \frac1L\sum_\ell
  \|\widehat A_{\ell}^{(s+1)}-A_\ell^{(s+1)}\|_F^2
  +
  \|\widehat A_{\ell}^{\rightarrow,(s+1)}
     -A_\ell^{\rightarrow,(s+1)}\|_F^2.
\]
This is the practical analogue of functorial analogy in the paper: not only
`source + f = target`, but a map between collections of related thought
vectors and their simplicial structure.

More precisely, the desired object is a functor with additional filtered
topological structure.  At a fixed radius, \(P\) induces a soft simplicial map;
across radii it should respect inclusions; on chains it should approximately
commute with the boundary; and over the whole radius sweep it should define an
approximate morphism of persistence modules:
\[
  \begin{CD}
  H_p(K_s(\rho_1)) @>>> \cdots @>>> H_p(K_s(\rho_L))\\
  @V{P_{*,1}}VV                 @. @V{P_{*,L}}VV\\
  H_p(K_{s+1}(\rho_1)) @>>> \cdots @>>> H_p(K_{s+1}(\rho_L)).
  \end{CD}
\]
The directed version carries a noncommutative orientation: reversing arrows can
change the ordered simplex set and therefore the induced chain map.  The
implementation therefore tracks more than a categorical correspondence.  It
tracks whether the functor-like map preserves the local topology, the
filtration order, directed cycle flux, and the toric skew structure that makes
reasoning flow noncommutative.

## Implementation Tasks

Completed in this branch:

1. `src/toricgt/random_order_lm.py`
   - GraphCG chart is used for relation topology and step-local topology.
   - Logs chart dimension and chart energy.
   - Exposes map residual metrics:
     `train/analogy_step_analogical_map_loss`,
     `train/analogy_step_directed_map_loss`,
     `train/analogy_step_transport_entropy`.
2. `src/toricgt/topological_reasoning.py`
   - Builds directed local complexes over reasoning windows.
   - Computes soft maps between consecutive complexes.
   - Returns differentiable map residuals and NumPy audit summaries.
3. `scripts/evaluate_reasoning_geometry_suite.py`
   - Adds toric phase projection paths.
   - Adds real-model 3D toric phase/simplicial trajectory plots.
   - Adds map-loss heatmaps and summary metrics.
4. `scripts/generate_paper_figures.py`
   - Adds manuscript figures for the GraphCG-chart topology stack and toric
     phase simplicial trajectories.

Next-iteration paper extension:

- The same toric chart should support affine Coxeter and braid-family tasks
  when a root datum is supplied. Character/cocharacter lattices give the
  coordinate lattice, roots define reflection walls, translated root walls
  define affine Weyl/Coxeter actions, and ordered wall crossings define braid
  actions.
- The noncommutative torus phase projection should be treated as a finite
  diagnostic shadow of an irrational foliation on a commutative torus. Smooth
  projected phase leaves are desirable between verified tropical, braid, or
  affine-Coxeter wall crossings; high leaf residual indicates incoherent
  toric memory.

## Training Policy

The auxiliary weights should remain small compared with BPB.  The intended
schedule is:

- early BPB capture: GraphCG/topology near zero;
- stable negative BPB slope: GraphCG basis begins learning;
- post-stabilization: directed topology and map residuals ramp in;
- evaluation/test-time scaling: multiple random orders and GFlowNet branches
  are compared by BPB, K proxies, Dirichlet energy, toric recurrence, and map
  residuals.

If BPB worsens while map residuals improve, reduce topology weight first.  If
BPB improves but Dirichlet energy and map residuals stagnate, increase only the
GraphCG basis weight or batch size before increasing topology pressure.

## Acceptance Criteria

1. Unit tests pass for random-order LM and topology utilities.
2. W&B logs finite GraphCG, analogy, topology, and map metrics.
3. Periodic analyses write:
   - `*_toric_phase_simplicial_trajectory.png`;
   - `*_directed_filtration.png`;
   - `*_noncommutative_heatmaps.png`;
   - `*_step_radius_hierarchy.png`.
4. `analogical_map_loss` and `directed_map_loss` are finite and should drift
   downward after the GraphCG chart stabilizes.
5. Toric phase recurrence remains nonzero and pseudo-periodic rather than
   collapsing into a small set of phase bins.
