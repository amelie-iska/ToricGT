from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_oai_competition_bpb.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_oai_competition_bpb", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_native_oai_evaluator_detects_compact_seq4096_payload() -> None:
    module = load_module()
    payload = {"step": 123, "model": {"tok_emb.weight": torch.zeros(8, 4)}}

    assert module.is_compact_seq4096_payload(payload) is True
    assert module.is_compact_seq4096_payload({"config": {}, "model": {"other": torch.zeros(1)}}) is False


def test_compact_seq4096_batches_map_to_val_max_sequences() -> None:
    module = load_module()
    args = Namespace(val_max_sequences=0, batches=3)

    assert module.compact_seq4096_max_sequences(args, seq_len=256, val_batch_size=1024) == 12

    args = Namespace(val_max_sequences=7, batches=3)
    assert module.compact_seq4096_max_sequences(args, seq_len=256, val_batch_size=1024) == 7
