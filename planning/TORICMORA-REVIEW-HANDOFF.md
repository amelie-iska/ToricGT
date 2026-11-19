# ToricMora review before training

The November 19 preparation iteration stops at owner review. No earlier training goal authorizes a launch on this candidate.

## Integration baseline

Continue from ToricMora. Its public branch at `15d5c51441d0c954f11ad54ccc675e3e8b015a8a` already descends from `toricblm-data` at `8b07b13` and retains the TokenGT and ToricBLM routes. The private Mora-3 checkout at `66a1d4f` serves as behavioral reference. The current symbolic work adds an opt-in ordered forest grammar with ancestor-only proposal visibility, exact simplex verification and verified forest voting, native prepared-stream recovery, and byte-accounting checks. Existing postfix search and existing trainer entry points keep their defaults. See the [readiness audit](TORICMORA-TRAINING-READINESS.md) for tests and capability limits.

## Iteration plan

- [x] Verify the integration base and preserve the source checkouts and existing training routes.
- [x] Add and verify ordered symbolic forest generation and the no-training curation and launch entry points.
- [x] Stage a reviewable public arithmetic candidate with separate train, validation and test roles. Keep its data artifacts on Iska.
- [ ] Owner reviews, filters and approves the exact input manifest. Rebuild the candidate after edits and use the new pin.
- [ ] Only after approval, run two CUDA epochs with restart after epoch one and verify the checkpoint and validation receipts.

The preparation here covers symbolic and general-text work. No biomedical curation or specialized biological generation work belongs to this review branch.

## Data ready for review

The first candidate uses the publisher's MIT-licensed GSM8K release at commit `3101c7d5072418e28b9008a6636bde82a006892c`. Source bytes and revision are pinned in the importer. Each training graph preserves the question, source reasoning lines and final answer as ordered records. The data conversion does not create new solutions. Test records participate in the held-out metadata audit and their graphs are excluded. A canonical question identity keeps variants together. The candidate contains **5,961 training**, **788 validation**, **2,043 test-role**, and **0 retrieval** records after the symbolic audit. All 1,319 publisher test records retain test-only membership. [Official source and license](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c)

The import, reviewable JSONL, admission envelope, and any further filtered packages are separate files. Edit a copy of the reviewable JSONL. Retain the official test metadata so the next bundle still excludes test-linked records. Then repackage into a fresh directory.

```sh
PYTHONPATH=src python scripts/prepare_symbolic_stream.py \
  --records "$FILTERED_ROWS" --output "$NEW_FILTERED_BUNDLE"
```

## Training controls

[config.yml](../config.yml) documents the native graph architecture, AdamW schedule, batch shape and accumulation, optional embedding-space losses, and runtime settings. `config.yaml` is an alias for the same file. The template leaves the manifest, digest and experiment directory empty.

- `scripts/run_toricmora_reviewed.sh` validates the recipe and exits.
- `scripts/run_toricmora_reviewed.sh plan --config "$REVIEWED_CONFIG"` verifies the supplied manifest, calculates exact epoch boundaries and prints the launch receipt without creating output.
- After owner approval, `scripts/run_toricmora_reviewed.sh train --config "$REVIEWED_CONFIG" --approve-manifest-sha256 "$APPROVED_SHA256"` runs the two segments under the existing GPU lease. It preserves the config and plan, uses separate checkpoint directories, and verifies the final record counts and checkpoint pins. The native graph objective reports masked MSE. The causal-language evaluator has a separate compatible checkpoint contract.

The isolated review branch contains source and tests. The source checkouts, local-only branch history, datasets, model weights, and experiment artifacts stay on Iska.
