# Anticipative Reasoning-Trajectory Memory

## Goal

ToricGT should learn to consult prior graph-of-thought trajectories without
turning the Parameter-Golf model into a large retrieval stack.  The memory
system stores compact summaries of completed reasoning trajectories and trains
an anticipative retrieval head that predicts which stored trajectories are
likely to help a new problem.  Later training phases can expose retrieved
trajectories through the long context supported by tropical ring attention.

This is a training and inference-time-scaling layer.  It is deliberately
separate from the byte-counted dense contest artifact unless a final deployment
decision explicitly includes it.

## Memory Object

For a completed reasoning trajectory

\[
\tau=(h_1,\ldots,h_T), \qquad h_t\in\mathbb R^d,
\]

the memory key is a normalized summary

\[
k(\tau)=\operatorname{norm}\big[
\bar h,\; h_T-h_1,\; \operatorname{speed}(\tau),\;
\operatorname{curvature}(\tau),\; \rho_{\mathrm{VR}}(\tau),\;
\phi_\theta(\tau),\;\phi_\beta(\tau),\;q(\tau)
\big],
\]

where \(\bar h\) is the mean hidden state, \(h_T-h_1\) is the net reasoning
displacement, \(\rho_{\mathrm{VR}}\) is a local Vietoris-Rips radius/density
summary, \(\phi_\theta,\phi_\beta\) are noncommutative toric phase-shadow
summaries, and \(q(\tau)\) is a quality proxy such as negative local NLL or
verifier reward.

The memory value stores the source metadata, optional text or graph projection,
terminal solution span, topology audit, toric audit, and GFlowNet branch score.

## Retrieval Score

Given current state \(s\) and memory item \(i\), the learned head predicts

\[
r_\psi(s,i)=
\frac{\langle Q_\psi(s),K_\psi(k_i)\rangle}{\tau_r}.
\]

The in-batch teacher used during training is

\[
\tilde r(s,i)=
\lambda_G\langle g_s,g_i\rangle
+\lambda_\Theta\langle \phi_s,\phi_i\rangle
-\lambda_T d_{\mathrm{top}}(s,i)
+q_i,
\]

where \(g\) are GraphCG chart coordinates, \(\phi\) are toric phase summaries,
and \(d_{\mathrm{top}}\) compares local topology summaries.  The loss is

\[
\mathcal L_{\mathrm{mem}} =
\operatorname{CE}(\operatorname{softmax} r_\psi,\arg\max_i\tilde r)
+ \eta\operatorname{KL}(\operatorname{softmax}\tilde r\Vert
\operatorname{softmax}r_\psi)
+ \xi\|\hat q-q\|_2^2.
\]

This gives the model a retrieval policy before the offline database is used in
full training.

## Prefix and Competition Safety

For Parameter Golf, retrieval must not read target validation bytes.  The safe
policy is:

- training memory can use completed training trajectories only;
- validation/test-time memory can use only a fixed offline library generated
  before the scored stream or legal score-before-update state;
- retrieved graph/text summaries must be injected into context before scoring
  only when they are independent of the target byte being scored;
- the auxiliary retrieval head is excluded from packed-artifact accounting until
  the final artifact explicitly uses it.

## Implementation Status

- `src/toricgt/trajectory_memory.py` implements CPU summaries, JSONL index
  save/load/search, and `TrajectoryRetrievalHead`.
- `DenseRandomOrderToricLM` can instantiate the head with
  `use_trajectory_memory_head=true`.
- `scripts/train_parameter_golf_random_order.py` logs
  `train/trajectory_memory_*` metrics and supports phase-controlled
  `trajectory_memory_loss_weight`.
- `config/train.parameter_golf_random_order_dense_valmix35_from1000.yaml`
  enables the head but keeps the loss at zero during BPB-first recovery.  The
  loss turns on only in later GFlowNet/GraphCG/topology phases.

## Next Training Step

After the base BPB descent stabilizes, build a small training memory from
completed branches with low answer BPB and high topology/toric consistency.
Then enable retrieved-context augmentation for graph-heavy batches:

1. Build/update `data/trajectory_memory/train_memory.jsonl` from completed
   training branches.
2. For each new graph-heavy batch, retrieve \(K=4\) to \(8\) analogical
   trajectories by the learned head plus cosine fallback.
3. Serialize the retrieved graph summaries into the long ring-attention context.
4. Train with a small retrieval alignment weight and monitor:
   `trajectory_memory_recall1`, `trajectory_memory_entropy`,
   `trajectory_memory_score_gap`, validation BPB, answer BPB, and topology
   transfer quality.

## Failure Modes

- Low retrieval entropy with poor recall means premature collapse to a single
  memory pattern.
- High recall but worse BPB means retrieved context is distracting the byte
  model and should be restricted to graph-heavy or inference-scaling batches.
- Strong topology similarity but weak answer quality means the teacher score
  should increase the quality term or include verifier reward.
- Useful retrieval with flat BPB is still valuable for inference-time scaling,
  but not for the contest artifact unless it improves byte likelihood under the
  rules.
