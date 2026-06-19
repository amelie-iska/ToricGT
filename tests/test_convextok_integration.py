from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PG_ROOT = ROOT / "amelie-iska" / "parameter-golf"
if str(PG_ROOT) not in sys.path:
    sys.path.insert(0, str(PG_ROOT))

from convextok import ConvexTokTokenizer, analyze_tokenizer_regret, tokenisation_dag_payload, train_convextok


def test_convextok_lp_roundtrip_and_regret(tmp_path: Path) -> None:
    texts = [
        "abc abd abe abc abd",
        "tropical tokenisation is a min plus shortest path over a byte-boundary dag",
        "toric vocabulary faces expose active path changes",
    ]
    result = train_convextok(
        texts,
        vocab_size=320,
        rounding="det",
        max_candidates=96,
        max_docs=3,
        max_doc_bytes=160,
    )
    assert result.lp_success
    assert result.selected_candidate_count > 0
    tok = result.tokenizer
    encoded = tok.encode(texts[0])
    assert encoded
    assert tok.decode(encoded) == texts[0]

    path = tmp_path / "toy.convextok.json"
    tok.save_json(path)
    loaded = ConvexTokTokenizer.load_json(path)
    assert loaded.decode(loaded.encode(texts[1])) == texts[1]

    metrics = analyze_tokenizer_regret(texts, loaded, max_candidates=64, max_doc_bytes=160)
    assert metrics["tokenizer_regret/lp_lower_bound_tokens"] <= metrics["tokenizer_regret/convextok_path_tokens"]
    assert metrics["tokenizer_regret/convextok_gap_ratio"] >= 1.0
    assert metrics["tokenizer_toric/lp_face_mass"] >= 0.0


def test_convextok_tokenisation_dag_payload() -> None:
    texts = ["token graph token graph", "graph token graph token"]
    tok = train_convextok(texts, vocab_size=288, max_candidates=32, max_docs=2).tokenizer
    payload = tokenisation_dag_payload(tok, texts[0], max_bytes=64)
    assert payload["schema"] == "toricgt.convextok_tokenisation_dag.v1"
    assert payload["nodes"]
    assert payload["edges"]
    assert any(edge["type"] == "free_byte_edge" for edge in payload["edges"])
    assert any(edge.get("selected") for edge in payload["edges"])
    json.dumps(payload)

