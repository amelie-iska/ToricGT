# ConvexTok, Tropical DP, and Toric Embedding Integration Plan

This plan is the implementation contract for adding ConvexTok-style tokenisation to the ToricGT OAI baseline adaptation.  The objective is to lower BPB while preserving the project direction: all FineWeb data is graphified, the OAI Parameter-Golf baseline remains the BPB-scored model, and graph-output flattening is used only for OAI FineWeb scoring.

The work must be implemented in full, tested, documented, and wired into the 1K-step BPB gate loop.  The current training loop should not be interrupted until the new tokenizer path is ready for a restart.

## Mathematical Framing

ConvexTok is a tokenisation graph optimization.  For a byte string, vertices are byte-boundary positions and directed edges are candidate tokens.  Free one-byte edges guarantee feasibility; priced edges are candidate substrings selected by vocabulary colours.  The integer problem selects at most `K` colours and then routes a shortest path through each document.  The LP relaxation gives a lower bound on token count, while deterministic/bias/integer rounding produces candidate vocabularies.

For ToricGT this is not just preprocessing:

1. **Tropical dynamic programming.**  Encoding with a fixed ConvexTok vocabulary is a min-plus shortest-path computation on a byte-boundary DAG.  The selected token path, active predecessor, top-two margin, and path entropy are tropical attention analogues and should be logged.
2. **Boolean-circuit reasoning and expressivity.**  The tokenisation DAG gives a finite Boolean/min-plus circuit: token colour selection gates, edge availability gates, and path feasibility gates.  These are the same kinds of finite circuits our tropical attention and TokenGT graphification are meant to expose.
3. **Toric embedding.**  Candidate-token incidence vectors are exponent vectors.  LP scores and rounded vocabularies define points/faces in a vocabulary polytope.  The active shortest-path regions form a normal-fan shadow; token path changes are wall crossings.  These quantities belong in the existing toric/tropical metrics rather than a generic tokenizer report.
4. **OOD generalization.**  A tokenizer with lower regret against the LP bound gives the model fewer brittle segmentation artifacts.  The toric/tropical path diagnostics should tell us whether BPB is tokenizer-bound, model-bound, or auxiliary-loss-bound.

## Checklist

- [x] Add a reusable ConvexTok implementation with exact sparse LP solving on a configured tokenizer-training sample.
- [x] Support Det, Bias, and Int rounding from LP colour variables.
- [x] Implement exact shortest-path encoding and decoding for the rounded vocabulary.
- [x] Save/load ConvexTok tokenizer JSON with byte tokens, special tokens, LP metadata, token scores, and graph-feature metadata.
- [x] Export ConvexTok FineWeb shards from the official matched document cache when available.
- [x] Provide a documented local smoke mode that builds tiny shards from a bounded text sample for tests only.
- [x] Add tokenizer-regret analysis: current path length, ConvexTok path length, LP lower bound, gap ratios, vocabulary utilization, tropical margins, active path entropy, and toric vocabulary-face statistics.
- [x] Add tokenisation DAG graph payload export with byte-boundary nodes and free/priced token edges.
- [x] Teach `train_gpt.py` to load `.convextok.json` alongside SentencePiece `.model`.
- [x] Make BPB byte accounting tokenizer-generic and exact for ConvexTok byte-string tokens.
- [x] Add optional ConvexTok DAG feature channels to the first-class TokenGT structural embedding.
- [x] Log `tokenizer_regret/*`, `tokenizer_tropical/*`, and `tokenizer_toric/*` metrics to W&B/review artifacts.
- [x] Update the 1K-step campaign loop with ConvexTok profiles: `convextok2048_det_tropical_toric_bpb` and `convextok2048_bias_ood_probe`, with 1024 tokenizer configs available for comparison.
- [x] Ensure artifact-size reporting still measures int8+zlib and verifies the 16MB cap.
- [x] Add unit tests for LP training, encoding/decoding, regret analysis, and graph payload generation.
- [x] Run smoke tests in the `tokengt` environment.
- [x] Run a GPU mini-train smoke test after implementation and before restarting the main loop.
- [x] Update docs and README with the tokenizer, tropical DP, toric embedding, and BPB-loop changes.
- [ ] Restart the 1K-step BPB campaign with the ConvexTok-2048 Det profile once the full path is ready.

## Implementation Decisions

1. The ConvexTok LP will be exact for the configured tokenizer-training sample using SciPy/HiGHS sparse linear programming.  The full FineWeb corpus is retokenized with the vocabulary learned from that sample, matching ordinary tokenizer-training practice.
2. The selected vocabulary always includes special tokens and all 256 byte tokens.  The `vocab_size` budget counts the total deployable vocabulary, so `K = vocab_size - special_count - 256` priced token colours.
3. Token costs default to one token per edge.  LP objective minimizes total routed edge count over the sample.  Bias rounding divides LP colour mass by byte length to favor compact edge reuse.
4. Encoding is an exact min-plus DP over byte positions.  It records active selected edges and a top-two margin when requested.
5. ConvexTok byte accounting is exact by construction: the scored byte count for a target token is the stored byte length for that token.  No SentencePiece metaspace correction is used.
6. Tokenisation DAG features are first-class inputs only when `CONVEXTOK_DAG_FEATURES=1` and the loaded tokenizer has token-score metadata.  They add compact embeddings/projections rather than changing the main vocabulary logits.
7. The review loop will treat tokenizer regret as a separate family.  It must not collapse all advanced metrics into one bin: tokenizer-bound, tropical-path instability, toric-face instability, auxiliary-gradient conflict, and model underfitting are separate hypotheses.

## Tests Required Before Restart

1. Tiny LP example builds a vocabulary and encodes/decode roundtrips.
2. Deterministic/Bias/Int rounding all produce valid tokenizers.
3. Regret metrics are finite and LP lower bound is no larger than the rounded path length.
4. Tokenisation DAG export contains byte-boundary nodes, free byte edges, priced candidate edges, LP scores, and selected-path annotations.
5. `train_gpt.py` accepts `.convextok.json` and computes train/val BPB without SentencePiece assumptions.
6. A 2-step GPU smoke run with `VOCAB_SIZE=2048` and ConvexTok data completes without OOM and reports tokenizer metrics.
