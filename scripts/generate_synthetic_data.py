#!/usr/bin/env python3
"""Generate synthetic ToricGT curriculum records."""

from __future__ import annotations

import argparse

from toricgt.synthetic import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic/toricgt_synthetic.jsonl")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    write_jsonl(args.output, count=args.count, seed=args.seed)
    print({"output": args.output, "count": args.count, "seed": args.seed})


if __name__ == "__main__":
    main()
