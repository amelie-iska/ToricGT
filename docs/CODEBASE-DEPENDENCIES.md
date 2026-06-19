# Codebase Dependencies

This repository treats runtime code dependencies as tracked git links or source
files, while keeping local credentials, generated runs, checkpoints, and data
artifacts out of git.

## Tracked Dependency Repositories

The following nested repositories are first-class dependencies and should be
cloned through `git submodule update --init --recursive`:

- `amelie-iska/parameter-golf`: OpenAI Parameter Golf baseline adaptation used
  by the active BPB training path.  The tracked branch is `oai-advanced`.
- `external/gflownet`: reference GFlowNet implementation used for comparison
  and method alignment.
- `external/Forest-of-Thought`: Forest-of-Thought reference implementation used
  for the embedding-space FoT training adaptation.

## Local Reference Clones

Other directories under `amelie-iska/` may exist on the development machine as
research/reference checkouts, but they are intentionally ignored unless they
become runtime dependencies.  If a reference checkout becomes required by a
script, training entrypoint, or import path, promote it explicitly as a
submodule and document the exact branch and commit.

## Secret And Artifact Policy

Do not commit local credentials or generated training state.  In particular:

- `keys.txt` is ignored and must remain untracked.
- `.env`, `.env.*`, `*.pem`, and `*.key` are ignored.
- `checkpoints/`, `runs/`, `training_notes/`, `wandb/`, `outputs/`, and `logs/`
  are ignored generated artifacts.

Before pushing dependency changes, check:

```bash
git status --short --branch
git ls-files keys.txt .env '.env.*' '*.pem' '*.key'
git submodule status --recursive
```

The first command should show only intentional source/submodule changes, the
second should print nothing, and the third should show the tracked dependency
commits.
