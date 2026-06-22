# ToricBLM ConvexTok-8192 Biomedical Restart Plan

## Objective

Stop the current ToricBLM run and restart from step 0 with a larger fresh ConvexTok vocabulary for a universal-modality biomedical de novo design and reasoning agent. The tokenizer must remain ConvexTok, not BPE. The first budget is 8192 total ids. Layer count remains 9 unless later evidence shows that a layer change helps without exceeding about 170M parameters.

## Constraints

- Project root remains ToricGT.
- Dataset source stays under `/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data`, but all code/configuration lives in ToricGT.
- Current 2048 token shards may be removed to make room for the 8192 retokenized shards.
- Fresh 8192 vocabulary means no preservation or copying of the old 2048 priced-token vocabulary.
- muP shape files must be generated with the same vocabulary size as training.
- The initial target model must stay below about 170M parameters.

## Chosen Initial Configuration

- Tokenizer: fresh ConvexTok-8192 deterministic rounding.
- Special/byte fallback ids: unchanged ConvexTok scheme, ids `0..259`.
- Priced-token budget: `8192 - 260 = 7932`.
- Biomedical reserve: 512 exact control/modality tokens from `convextok_biomed_universal_modality_seed.jsonl`.
- Corpus LP budget: remaining priced-token budget selected from the curated training Parquet sample.
- Corpus token selection: frequency-ranked ConvexTok candidates for this restart. Exact LP attempts at 512/9000, 384/8500, and 256/5000 were too slow for the local restart loop. The tokenizer is still a fresh ConvexTok shortest-path vocabulary, not BPE, and the 512-token reserve is preserved.
- Model: 9 layers, width 1536, 12 attention heads, 6 KV heads, MLP multiplier 2.
- Target parameter count from generated muP config: `169,698,927`.
- Training batch tokens initially reduced to `196608` to compensate for the larger output vocabulary.

## Validation Before Full Export

1. Train the fresh ConvexTok-8192 tokenizer.
2. Compare token counts against the current ConvexTok-2048 tokenizer on:
   - curated training text sample;
   - biomedical/universal-modality tokenizer seed text.
3. Confirm the tokenizer JSON loads and `VOCAB_SIZE=8192` matches.
4. Remove only the old 2048 `.bin` token shards, preserving source Parquet and metadata.
5. Export full train and validation `.bin` shards with the fresh tokenizer.
6. Launch the muP ToricBLM run from step 0 with the 8192 config.

## Completed Tokenizer Evidence

The fresh ConvexTok-8192 tokenizer was built with frequency-ranked corpus
selection after exact LP attempts proved too slow for the local restart loop.
It is still a new ConvexTok vocabulary: it does not reuse or preserve the old
ConvexTok-2048 vocabulary.  The tokenizer JSON reports:

- `vocab_size`: 8192
- priced tokens: 7932
- unique seed reserve tokens actually present: 323 from the 512-token reserve
  target, with duplicates naturally collapsed by byte string
- corpus priced tokens: 7609
- selection: frequency-ranked ConvexTok candidates
- base tokenizer reused: false

Comparison against the previous ConvexTok-2048 tokenizer on
`outputs/convextok8192_biomed_det_comparison.json`:

| Corpus | Old tokens | New tokens | Token delta | Old bytes/token | New bytes/token |
|---|---:|---:|---:|---:|---:|
| 1024 curated train texts | 993,945 | 790,185 | -20.50% | 2.466 | 3.102 |
| biomedical/control seed | 3,010 | 669 | -77.77% | 1.587 | 7.142 |

This is enough to justify the 8192 restart as a real tokenizer-capacity test:
it reduces sequence length on current curated text while sharply improving
future biomedical/control syntax coverage.  The next decision is empirical
training quality: if BPB improves early and memory remains stable, later vocab
experiments may consider 12288 or 16384 with the same 512-token reserve rule.

## Decision Rule For Later Vocab Budgets

Use 8192 for this restart. After export and an initial run, decide whether to move to 12288 or 16384 only if:

- tokenized sequence length drops enough to offset larger softmax cost;
- validation/train BPB improves early;
- compressed artifact size and memory stay within practical limits;
- biomedical modality/control coverage improves without harming FineWeb BPB.
