#!/usr/bin/env python3
"""Convert the pinned official GSM8K release into symbolic curation input."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

from toricgt.toricmora_symbolic_splits import GSM8K


REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
SPLIT_PINS = {
    "train": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
    "test": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
}


def convert_row(row: dict, split: str, index: int) -> dict:
    question = row["question"].strip()
    explanation, separator, final_answer = row["answer"].rpartition("####")
    final_answer = final_answer.strip()
    steps = [step.strip() for step in explanation.splitlines() if step.strip()]
    if not (question and separator and final_answer and steps):
        raise ValueError("GSM8K row is missing its question, explanation or marked answer")
    canonical_question = " ".join(unicodedata.normalize("NFC", question).split())
    identity = hashlib.sha256(canonical_question.encode("utf-8")).hexdigest()
    result = {"record": {
        "record_id": f"gsm8k:{REVISION[:12]}:{split}:{index:05d}",
        "source": GSM8K, "source_id": identity,
        "canonical_question": canonical_question, "canonical_answer": final_answer,
        "declared_holdout": "test" if split == "test" else None,
        "retrieval_allowed": False,
    }}
    if split == "train":
        text = [("question", question), *(("step", step) for step in steps), ("answer", final_answer)]
        nodes = [{"id": str(i), "type": role, "text": value} for i, (role, value) in enumerate(text)]
        edges = [{"source": str(i), "target": str(i + 1), "type": "ordered_reasoning"} for i in range(len(text) - 1)]
        result["graph"] = {"nodes": nodes, "edges": edges}
    return result


def read_official(path: Path, split: str) -> list[dict]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SPLIT_PINS[split]:
        raise ValueError(f"{split} bytes do not match the pinned official GSM8K release")
    return [convert_row(json.loads(line), split, i) for i, line in enumerate(raw.splitlines(), start=1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        training = read_official(args.train, "train")
        held_out = read_official(args.test, "test")
        with args.output.open("x", encoding="utf-8") as dst:
            for row in (*training, *held_out):
                dst.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    except (OSError, TypeError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({"revision": REVISION, "source_train": len(training), "source_test": len(held_out)}))


if __name__ == "__main__":
    main()
