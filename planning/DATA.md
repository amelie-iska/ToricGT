# ToricGT Data Research, Curation, and Segmentation Plan

Author: Amelie Schreiber

Date: May 25, 2026

## Objective

Build a local, reproducible graph-reasoning data corpus for ToricGT with:

- chain-of-thought, tree-of-thought, and graph-of-thought reasoning traces;
- math and algorithmic/programming tasks;
- OpenAI-origin GSM8K coverage;
- Hebrew morphology and Jewish Hebrew/rabbinic text coverage;
- deterministic leakage-resistant 80/10/10 train/validation/test segmentation;
- Parquet outputs ready for training and, after license review, Hugging Face upload.

The resulting curation is graph-first: every raw row is normalized into a record with question/prompt fields, answer/solution fields, reasoning text, metadata, stable hashes, estimated token counts, and a deterministic split.

## Source Research Results

The following datasets were inspected through Hugging Face search, dataset cards, repository file metadata, and `datasets` schema discovery in the Conda `tokengt` environment.

| Dataset | Selected Config / Loader | Splits / Rows | Size Observed | License | Role |
|---|---|---:|---:|---|---|
| `AI-MO/NuminaMath-CoT` | `default` via `datasets` | train 859,494; test 100 | 5 train parquet shards, ~1.23GB compressed | Apache-2.0 | large math CoT |
| `open-r1/OpenR1-Math-220k` | `all` via `datasets` | train 225,129 | all/train shards ~4.2GB compressed | Apache-2.0 | multi-trace math reasoning |
| `open-r1/codeforces-cots` | `solutions_w_editorials_py_decontaminated` via `datasets` | train 9,796 | 3 parquet shards, ~466MB compressed | CC-BY-4.0 | algorithmic/programming CoT |
| `openai/gsm8k` | `main` via `datasets` | train 7,473; test 1,319 | ~2.7MB compressed | MIT | OpenAI math word problems |
| `EleutherAI/hendrycks_math` | all 7 subject configs via `datasets` | train/test per subject, ~12.5k total | ~4.9MB compressed | MIT | competition math |
| `HuggingFaceH4/MATH-500` | `default` via `datasets` | test 500 | ~447KB JSONL | dataset card | OpenAI verifier subset benchmark |
| `terrycraddock/Tree_Of_Thoughts_BASE_24k` | raw JSON file | train 24,731 | `Tree_Of_Thoughs_qa_base_24k.json`, ~108MB | Apache-2.0 | explicit ToT |
| `gss1147/Got_Math_500K` | raw JSONL file | ~500k target rows | `Got_Math_500K.jsonl`, ~490MB | Apache-2.0 | explicit GoT / graph-of-thought math |
| `unimorph/universal_morphologies` | raw UniMorph Hebrew URL fallback | Hebrew table from UniMorph GitHub | small text table | CC-BY-SA-3.0 | Hebrew lemma/inflection morphology |
| `Sefaria/Rabbinic-Hebrew-English-Pairs` | `default` via `datasets` | train 3,708 | `data/benchmark.json`, ~8.3MB | CC-BY-4.0 | rabbinic Hebrew/Aramaic-English parallel text |
| `Sefaria/hebrew_library` | `default` via `datasets` | train 3,549,020 | `data.jsonl`, ~7.7GB | GPL-3.0 | Sefaria Hebrew Jewish text library |
| `Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b` | `stage1`, `stage2` via `datasets` | ~435k target rows | JSON/Parquet Hub exports | CC-BY-4.0 | OpenAI gpt-oss-120b frontier reasoning distillation |
| `Jackrong/GPT-OSS-120B-Distilled-Reasoning-math` | `default` via `datasets` | 1K-10K | JSONL | Apache-2.0 | explicit gpt-oss-120b native math reasoning |
| `reasoning-degeneration-dev/algorithmic-sft-training-data-v1` | `default` via `datasets` | train 63,000 | Parquet | MIT | deterministic procedural reasoning traces |
| `lamm-mit/graph-reasoning-messages-11K` | `default` via `datasets` | ~11k | Parquet | Apache-2.0 | graph-native reasoning conversations |
| `sequelbox/DAG-Reasoning-DeepSeek-R1-0528` | `default` via `datasets` | 1K-10K | CSV/Hub dataset | Apache-2.0 | DAG reasoning distilled from DeepSeek-R1-0528 |
| `Gryphe/Opus-4.6-Reasoning-24k` | `default` via `datasets` | ~24k | JSON | Apache-2.0 | frontier Opus 4.6 reasoning aggregate |

## Important Loader Notes

`unimorph/universal_morphologies` cannot be loaded through current `datasets` because the Hub repository contains an old dataset script and current `datasets` rejects dataset scripts. The curation code therefore downloads the Hebrew file directly from:

```text
https://raw.githubusercontent.com/unimorph/heb/master/heb
```

`gss1147/Got_Math_500K` does not expose features through the standard builder. The curation code downloads and streams its raw JSONL file from the Hub.

`terrycraddock/Tree_Of_Thoughts_BASE_24k` is a raw JSON file, not a conventional multi-shard dataset. The curation code downloads and parses that file directly.

The Hebrew/Jewish-text slice deliberately excludes Christian-branded or non-Jewish biblical-language datasets. It uses Sefaria and UniMorph Hebrew sources instead.

Niqqud policy: preserve pointed Hebrew whenever the upstream source provides it, prefer a same-consonant pointed variant when an equivalent field is present, and explicitly flag unpointed Hebrew rather than silently treating it as pointed text. The curation/enrichment code records `has_niqqud`, `has_hebrew_without_niqqud`, `hebrew_letters`, `niqqud_marks`, and `niqqud_per_hebrew_letter` in row quality flags; `scripts/enrich_hebrew_niqqud.py --update-metadata` also mirrors that payload into metadata when needed. A strict pointed-Hebrew subset can be produced with `scripts/enrich_hebrew_niqqud.py --drop-unpointed-hebrew`, but the default is annotation rather than deletion because much rabbinic and modern Hebrew source material is legitimately unpointed. Run `scripts/report_hebrew_niqqud.py --curated-dir data/curated` before upload. The current audit finds `3,564,756` Hebrew rows: `2,868` pointed upstream rows and `3,561,888` unpointed upstream rows. Training jobs that require niqqud-only Hebrew should filter out rows with `has_hebrew_without_niqqud=true`.

The gpt-oss source selection is anchored on OpenAI's open-weight reasoning models. OpenAI describes `gpt-oss-120b` and `gpt-oss-20b` as open-weight reasoning models under Apache-2.0; the curated text traces above use public Hugging Face datasets that attribute their teacher to `gpt-oss-120b`. The companion `Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b-Logprob` dataset is not used as primary text because its rows contain token IDs and logprobs rather than decoded text; keep it as an optional reward-model or GFlowNet sidecar after tokenizer alignment.

Several newer combined frontier-distillation datasets exist, including Opus/Kimi/GLM mixtures from May 2026, but some use `license:other`. They should remain in a review quarantine until their redistribution and training terms are clear.

## Normalized Record Schema

Every curated row is written with these fields:

- `record_id`: stable SHA-256 identifier;
- `dataset`: Hugging Face dataset id or synthetic source id;
- `config`: dataset config or loader variant;
- `source_split`: original upstream split;
- `source_index`: row index within source stream;
- `task_family`: one of `cot_math`, `tot`, `got_math`, `algorithmic_code`, `openai_math`, `competition_math`, `hebrew_morphology`, `rabbinic_parallel`, `jewish_hebrew_text`;
- `language`: primary language tag;
- `license`: dataset license string from research pass;
- `role`: short human role description;
- `question`: prompt/problem/source form;
- `answer`: final answer or target;
- `solution`: full solution, generation, or target text;
- `reasoning`: chain/tree/graph-of-thought text when separable;
- `metadata_json`: compact JSON metadata with source-specific fields;
- `text`: normalized training text view;
- `graph_json`: lightweight graph representation for graph-token conversion;
- `content_hash`: exact normalized-content hash;
- `group_hash`: leakage-control grouping key;
- `split`: deterministic split label;
- `estimated_tokens`: whitespace token estimate;
- `quality_flags_json`: parse and quality annotations.

The local curation pass does not apply viewpoint, topic, or safety-content censorship. Filtering is limited to loader failures, exact/near duplicate grouping, structural parse quality, source attribution, and license-driven exclusion. Potentially sensitive source domains should remain identifiable through `dataset`, `role`, `task_family`, `license`, and `metadata_json` so downstream experiments can explicitly choose whether to include or exclude them.

## Graph Conversion Policy

The initial curation emits a lightweight graph JSON for every row.

Base graph:

- a `problem` node for question/prompt/source;
- zero or more `reasoning_step` nodes split from the solution/reasoning field;
- an `answer` node;
- `depends_on` edges from step to previous step;
- `supports_answer` edge from final step or problem to answer.

Dataset-specific graph additions:

- Codeforces rows add `problem_statement`, `editorial`, `public_test`, and `code_solution` nodes when available.
- OpenR1 rows add one generation node per sampled reasoning trace when available.
- UniMorph rows add `lemma`, `form`, and `features` nodes.
- Sefaria rabbinic parallel rows add `ref`, `hebrew_or_aramaic`, `english`, and `category` nodes.
- Sefaria Hebrew library rows add `ref`, `category`, `version`, and `text` nodes when metadata is available.
- ToT/GoT rows preserve branch markers where available and tag records as `tot` or `got_math` for later richer conversion.
- gpt-oss rows map `input` to problem nodes and `<think>`/native reasoning traces to reasoning-step nodes.
- Algorithmic SFT rows preserve deterministic state-transition traces, task labels, algorithm labels, and JSON metadata as graph features.
- Graph-reasoning message rows parse user/assistant conversations into problem, thought/action, and answer nodes.
- DAG reasoning rows are tagged for explicit thought-graph and action-graph parsing.

This graph representation is intentionally conservative. It is enough for TokenGT pretraining and can be replaced by a richer parser later without redownloading sources.

## Leakage-Controlled Segmentation

Random record-level splitting is not used.

The splitter computes:

1. canonical normalized text;
2. exact `content_hash`;
3. task-family keys such as Codeforces problem id, OpenR1 uuid, Sefaria reference, or UniMorph lemma/form;
4. a stable SimHash prefix over word shingles.

The `group_hash` combines task-family key, content hash, and SimHash prefix. The split is assigned by hashing the group key:

- `train`: hash percentile `[0, 80)`;
- `validation`: `[80, 90)`;
- `test`: `[90, 100)`.

All rows with the same group key go to the same split. This is deterministic, parallel-safe, and does not require loading all rows in memory.

## Output Layout

```text
data/
├── curated/
│   ├── manifest.json
│   ├── split_report.json
│   ├── split_report.md
│   ├── all.parquet
│   ├── train.parquet
│   ├── validation.parquet
│   ├── test.parquet
│   └── by_dataset/
│       └── <safe_dataset_name>.parquet
└── raw/
    └── hf/
        └── downloaded raw JSON/JSONL fallback files
```

Large raw fallback files are stored under `data/raw/hf`. Standard `datasets` streams may also populate the Hugging Face cache outside the repo.

## Expected Scale

Approximate curated row count is about 5.8M records before exact filtering:

- NuminaMath: ~859k;
- OpenR1 all: ~225k;
- Got Math: ~500k;
- Codeforces: ~9.8k;
- GSM8K: ~8.8k;
- MATH: ~12.5k;
- ToT: ~24.7k;
- Sefaria Hebrew library: ~3.55M;
- Sefaria rabbinic parallel pairs: ~3.7k;
- UniMorph Hebrew: small morphology table.
- gpt-oss-120b text traces: ~435k plus a small explicit math distill;
- procedural algorithmic traces: 63k;
- graph/DAG/frontier Opus traces: roughly 35k-45k before exact inspection.

For pretraining, use the Chinchilla-style rule of thumb as a lower bound: roughly 20 high-quality training tokens per active parameter for one-epoch autoregressive training. That implies:

| Active Parameters | Minimum Text/Graph Tokens | Preferred Reasoning-Heavy Range |
|---:|---:|---:|
| 8M | 160M | 320M-800M |
| 15M | 300M | 600M-1.5B |
| 35M | 700M | 1.4B-3.5B |

The selected corpus should exceed the 35M lower bound after graph expansion, without requiring a full multi-epoch replay of every record on a 4090. Use streaming dataloaders, sequence/graph-token packing, per-family caps, and one-pass or low-epoch sampling. Treat inference-time scaling separately: GFlowNet rollouts, graph-of-thought branching, best-of-N, verifier selection, and budget forcing increase test-time compute without forcing the pretraining corpus itself to become unmanageably large.

## Commands

Run full curation:

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/curate_datasets.py \
  --output-dir data/curated \
  --raw-dir data/raw/hf \
  --chunk-size 20000 \
  --num-workers 16 \
  --normalize-batch-size 256
```

Run a bounded sample curation:

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/curate_datasets.py \
  --output-dir data/curated_sample \
  --raw-dir data/raw/hf \
  --max-records-per-source 1000 \
  --num-workers 4 \
  --normalize-batch-size 128
```

`aria2c` is useful for large direct downloads when available, but the current curated corpus is dominated by `datasets` streaming plus Python-side graph normalization, SimHash grouping, and Parquet writing. The curation script therefore exposes CPU normalization workers; use 8-16 workers on a 32-thread workstation and reduce the value if other CPU-heavy jobs are running.

Validate outputs:

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/inspect_curated_data.py \
  --curated-dir data/curated
```

## Publishing Notes

Before pushing parquet files to Hugging Face:

1. Re-authenticate locally with `huggingface-cli login`.
2. Re-check every dataset license.
3. Consider publishing only normalized metadata and graph conversion outputs where redistribution is allowed.
4. Keep source attribution columns in every row.
5. Do not publish records from any dataset whose terms prohibit redistribution or require additional gating.

## Scaling And Source References

The sizing policy follows:

- Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla): use approximately 20 training tokens per active parameter as the minimum one-epoch compute-efficient target for small dense or active-parameter MoE models.
- Snell et al., *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*: treat inference-time search, verifier scoring, and proposal selection as a separate compute budget that can improve reasoning without only scaling model size.
- Muennighoff et al., *s1: Simple Test-Time Scaling*: budget forcing can extract more reasoning from a small model after high-quality SFT, so ToricGT should report accuracy as a function of graph-of-thought rollout budget, not only as a function of parameters.
- OpenAI, *Introducing gpt-oss*: `gpt-oss-120b` and `gpt-oss-20b` are open-weight reasoning models under Apache-2.0 with full reasoning traces available from the open models; public gpt-oss-120b distillation datasets are therefore the preferred OpenAI-aligned frontier-reasoning source.

Links:

- https://arxiv.org/abs/2203.15556
- https://arxiv.org/abs/2408.03314
- https://arxiv.org/abs/2501.19393
- https://openai.com/index/introducing-gpt-oss/
