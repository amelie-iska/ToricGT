# DEC Conservative Reasoning Signals

This note records the comparison between `assets/1508.01166v2.pdf` and the
current ToricGT `oai` branch.

## Paper Reviewed

Mohamed, Hirani, and Samtaney, *Discrete exterior calculus discretization of
incompressible Navier-Stokes equations over surface simplicial meshes*
(`arXiv:1508.01166v2`) develops a conservative DEC discretization of
incompressible Navier-Stokes on simplicial surface meshes.

The useful ingredients for ToricGT are:

- rewrite vector calculus in exterior-calculus form using exterior derivative
  `d`, Hodge star `*`, wedge product, and interior product;
- discretize on primal/dual simplicial meshes so incidence identities are
  preserved by sparse boundary and coboundary matrices;
- handle the convective term algebraically through an interior-product operator
  and a combinatorial wedge product;
- evaluate physical fidelity by conserved mass, vorticity/circulation, and
  kinetic-energy behavior, not only by pointwise numerical error.

## Translation To ToricGT

ToricGT does not solve a PDE. The relevant object is a graph-of-thought hidden
trajectory expressed in the GraphCG concept chart. For each sampled reasoning
window and radius level, we already build a directed filtered flag complex:

```text
vertices:      hidden reasoning states
0-simplices:   reasoning states in a local window
1-simplices:   radius-thresholded hidden-state relations
2-simplices:   clique/flag triangles
direction:     temporal orientation + noncommutative toric skew
```

The DEC analogue is:

| DEC / NSE object | ToricGT analogue |
| --- | --- |
| velocity 1-form | directed adjacency skew `A^-> - (A^->)^T` |
| incompressibility | low row-sum/divergence of hidden reasoning flow |
| vorticity/circulation | signed local cycle pressure around nodes |
| kinetic energy | edge-weighted squared skew-flow energy |
| Hodge star | core-radius/mutual-reachability weighted edge metric |
| wedge/interior product | consistency between flow-vorticity wedge and local interior-product proxy |

The auxiliary residual is

```text
L_DEC =
  L_mass
  + 0.25 L_vorticity_drift
  + 0.10 L_kinetic_drift
  + 0.05 L_hodge_balance
  + 0.05 L_wedge_interior .
```

It is added only inside the existing step-local topology loss:

```text
L_step_topology <- L_step_topology + 0.15 L_DEC .
```

The total effect on BPB remains tiny because the top-level
`analogy_lattice_loss_weight` is already phased in at small values. The intent is
to reduce destructive high-frequency reasoning oscillations and improve
geometric stability without turning the model into a fluid solver.

## Implemented Metrics

Training logs the following W&B metrics:

- `train/analogy_step_dec_conservation_loss`
- `train/analogy_step_dec_mass_residual`
- `train/analogy_step_dec_vorticity_drift`
- `train/analogy_step_dec_kinetic_energy`
- `train/analogy_step_dec_kinetic_energy_drift`
- `train/analogy_step_dec_hodge_balance`
- `train/analogy_step_dec_wedge_interior_residual`

The periodic geometry suite reports corresponding `analysis/*` summary metrics
and adds DEC panels to:

- `geometry/topology/*_directed_filtration.png`
- `geometry/topology/*_step_radius_hierarchy.png`
- `reasoning_geometry_summary.json`

## Expected Healthy Behavior

Desired:

- mass/divergence residual decreases or remains small;
- vorticity drift remains bounded while directed cycle flux stays nonzero;
- kinetic energy changes smoothly instead of spiking at filtration transitions;
- Hodge balance does not dominate the topology loss;
- wedge/interior residual decreases after the GraphCG chart stabilizes.

Undesired:

- DEC loss falls to zero together with directed asymmetry and cycle flux, because
  that indicates collapse to a symmetric trivial complex;
- kinetic energy grows while BPB improves, because the model may be memorizing
  byte patterns with noisy hidden trajectories;
- Hodge balance dominates, because the core-radius metric is then fighting the
  BPB objective rather than serving as an audit.

## Resume Gate

Training should resume only after:

1. targeted tests pass;
2. the analysis suite runs on the latest checkpoint;
3. topology, trajectory, triangle, tetrahedron, toric-shadow, and DEC panels are
   visually reviewed;
4. the selected checkpoint and updated hyperparameters are recorded in
   `planning/METRICS.md` or the current run note.

