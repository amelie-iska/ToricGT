"""Native causal checkpoint proposer for the authored arithmetic benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .graph_dataset import collate_graph_items, fineweb_tokens_to_graph_json, graph_json_to_item
from .model import ToricTokenGT
from .toricmora_symbolic_search import EXEMPLAR, OPERATORS


def _verified_checkpoint(path, digest):
    with Path(path).open("rb") as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
        if checksum.hexdigest() != digest:
            raise ValueError("checkpoint SHA-256 mismatch")
        stream.seek(0)
        # The caller explicitly supplies a trusted local checkpoint.
        return torch.load(stream, map_location="cpu", weights_only=False)


def checkpoint_proposer(path, checkpoint_sha256, tokenizer_path, tokenizer_sha256, *, device):
    tokenizer_bytes = Path(tokenizer_path).read_bytes()
    if hashlib.sha256(tokenizer_bytes).hexdigest() != tokenizer_sha256:
        raise ValueError("checkpoint tokenizer SHA-256 mismatch")
    try:
        import sentencepiece
    except ImportError as exc:
        raise ValueError("checkpoint search requires the existing SentencePiece tokenizer runtime") from exc
    tokenizer = sentencepiece.SentencePieceProcessor(model_proto=tokenizer_bytes)
    snapshot = _verified_checkpoint(path, checkpoint_sha256)
    keys = {field.name for field in fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in snapshot["config"].items() if k in keys})
    if not (cfg.use_lm_head and cfg.use_lm_token_embeddings and cfg.use_causal_graph_attention):
        raise ValueError("checkpoint search requires native causal token logits")
    if tokenizer.get_piece_size() != cfg.lm_vocab_size:
        raise ValueError("the pinned tokenizer vocabulary differs from the checkpoint head")
    model = ToricTokenGT(cfg)
    model.load_state_dict(snapshot["model"], strict=True)
    model = model.to(device).eval()
    max_prompt_tokens = min(cfg.max_nodes, cfg.max_edges + 1)

    @torch.inference_mode()
    def propose(problem, seed, tree_id, prefix):
        prompt = json.dumps(
            {"example": EXEMPLAR, "trees": problem.trees, "targets": problem.targets,
             "root": tree_id, "operators_so_far": prefix},
            separators=(",", ":"),
        ) + " Next:"
        ids = tokenizer.encode(prompt, out_type=int)
        if not 0 < len(ids) <= max_prompt_tokens:
            raise ValueError("visible checkpoint prompt is empty or exceeds the token cap")
        if tokenizer.unk_id() in ids:
            raise ValueError("visible checkpoint prompt contains an unknown piece")
        candidate_ids = []
        for action in OPERATORS:
            with_operator = tokenizer.encode(prompt + " " + action, out_type=int)
            if with_operator[:-1] != ids or with_operator[-1] == tokenizer.unk_id():
                raise ValueError("each legal operator must encode as one extra token after the full prompt")
            candidate_ids.append(with_operator[-1])
        if len(set(candidate_ids)) != len(OPERATORS):
            raise ValueError("legal operators must encode as distinct continuation tokens")
        # The last ID is a structural sentinel. Only preceding IDs become model inputs.
        tokens = np.asarray([*ids, 0], dtype=np.int32)
        graph = fineweb_tokens_to_graph_json(tokens, max_nodes=cfg.max_nodes)
        batch, _ = collate_graph_items([graph_json_to_item(graph, cfg, lm_tokens=tokens, tokenizer=tokenizer)])
        batch = replace(batch, **{key: value.to(device) for key, value in vars(batch).items() if torch.is_tensor(value)})
        scores = model(batch)["lm_logits"][0, len(ids) - 1, candidate_ids].float().tolist()
        if not np.isfinite(scores).all():
            raise ValueError("checkpoint proposer produced non-finite operator scores")
        ranking = sorted(zip(OPERATORS, scores, strict=True), key=lambda choice: (-choice[1], choice[0]))
        return [action for action, _ in ranking], len(ids)

    return propose, {
        "checkpoint_sha256": checkpoint_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "prompt_token_cap_per_call": max_prompt_tokens,
        "device": str(device),
    }
