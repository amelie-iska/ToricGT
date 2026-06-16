#!/usr/bin/env python3
"""Render a dark-mode index for ToricGT output artifacts."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


CSS = """
:root{color-scheme:dark;--bg:#030712;--panel:#07111f;--text:#e8fbff;--muted:#91a8b7;--cyan:#37e8ff;--border:rgba(55,232,255,.28)}
body{margin:0;background:radial-gradient(circle at top left,#092238 0,var(--bg) 44rem);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1280px;margin:0 auto;padding:32px 24px 72px}.hero,.card{background:linear-gradient(180deg,rgba(11,23,40,.94),rgba(5,13,25,.98));border:1px solid var(--border);border-radius:8px;box-shadow:0 18px 50px rgba(0,0,0,.28)}
.hero{padding:24px;margin-bottom:18px}.grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}.card{padding:16px}
h1{margin:0 0 8px;font-size:29px;letter-spacing:0}h2{margin:0 0 10px;font-size:18px}p{color:var(--muted);line-height:1.5}a{color:var(--cyan);text-decoration:none}a:hover{text-decoration:underline}
.pill{display:inline-block;border:1px solid var(--border);border-radius:999px;padding:6px 10px;margin:4px 6px 0 0;background:rgba(55,232,255,.07);color:#dff9ff}.metric{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid rgba(145,168,183,.14);padding:6px 0}.metric span:first-child{color:var(--muted)}.metric span:last-child{text-align:right;overflow-wrap:anywhere;max-width:62%;}.missing{opacity:.62}.title{overflow-wrap:anywhere}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--index-path", default="")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def artifact_cards(root: Path) -> list[dict[str, Any]]:
    candidates = [
        (
            "Latest Branching Reasoning Simplex Report",
            root / "latest_branching_reasoning_trajectory_report" / "branching_reasoning_trajectory.html",
            root / "latest_branching_reasoning_trajectory_report" / "manifest.json",
            root / "latest_branching_reasoning_trajectory_report" / "html_screenshots" / "contact_sheet.png",
            "Long branch/merge reasoning trajectory, per-step simplex trees, analogical maps, and vectorized PH panels.",
        ),
        (
            "Latest Toric Embedding Report",
            root / "latest_toric_embedding_visual_report" / "index.html",
            root / "latest_toric_embedding_visual_report" / "manifest.json",
            root / "latest_toric_embedding_visual_report" / "html_screenshots" / "contact_sheet.png",
            "Tropical-to-toric embedding, staircases, resolutions, and CAS-backed toric geometry audits.",
        ),
    ]
    for path in sorted(root.glob("branching_reasoning_embedding_payload_*"))[-5:]:
        candidates.append(
            (
                f"Checkpoint Branching: {path.name.replace('branching_reasoning_', '')}",
                path / "branching_reasoning_trajectory.html",
                path / "manifest.json",
                path / "html_screenshots" / "contact_sheet.png",
                "Branch/simplex report rendered from a saved checkpoint embedding payload.",
            )
        )
    for path in sorted(root.glob("*/gudhi_persistence/index.html"))[-5:]:
        candidates.append(
            (
                f"GUDHI Persistence: {path.parents[1].name}",
                path,
                path.parent / "summary.json",
                path.parent / "html_screenshots" / "contact_sheet.png",
                "Exact GUDHI simplex trees, vectorized PH, and two-parameter persistence audits.",
            )
        )
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for title, html_path, manifest_path, screenshot_path, description in candidates:
        key = str(html_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        manifest = read_json(manifest_path)
        cards.append(
            {
                "title": title,
                "html": html_path,
                "manifest": manifest_path,
                "screenshot": screenshot_path,
                "description": description,
                "exists": html_path.exists(),
                "manifest_payload": manifest,
            }
        )
    return cards


def metric_rows(payload: dict[str, Any]) -> str:
    keys = ["schema", "source_mode", "nodes", "edges", "analogy_emitted", "ok_count", "error_count", "records"]
    rows = []
    for key in keys:
        if key in payload:
            value = payload[key]
            if isinstance(value, list):
                text = f"{len(value)} items"
            elif isinstance(value, dict):
                text = f"{len(value)} keys"
            else:
                text = str(value)
            if len(text) > 80:
                text = text[:77] + "..."
            rows.append(f"<div class='metric'><span>{html.escape(key)}</span><span>{html.escape(text)}</span></div>")
    return "".join(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    index_path = Path(args.index_path) if args.index_path else root / "index.html"
    cards = artifact_cards(root)
    html_cards = []
    for card in cards:
        exists = bool(card["exists"])
        cls = "card" if exists else "card missing"
        links = []
        if exists:
            links.append(f"<a class='pill' href='{html.escape(rel(root, card['html']))}'>open HTML</a>")
        if card["manifest"].exists():
            links.append(f"<a class='pill' href='{html.escape(rel(root, card['manifest']))}'>manifest/summary</a>")
        if card["screenshot"].exists():
            links.append(f"<a class='pill' href='{html.escape(rel(root, card['screenshot']))}'>contact sheet</a>")
        if not links:
            links.append("<span class='pill'>not generated</span>")
        html_cards.append(
            f"<section class='{cls}'><h2 class='title'>{html.escape(str(card['title']))}</h2>"
            f"<p>{html.escape(str(card['description']))}</p>"
            f"{metric_rows(card['manifest_payload'])}<div>{''.join(links)}</div></section>"
        )
    index_path.write_text(
        f"""<!doctype html><html><head><meta charset='utf-8'><title>ToricGT Outputs Index</title><style>{CSS}</style></head>
<body><main><section class='hero'><h1>ToricGT Outputs Index</h1>
<p>Central browser entry point for generated ToricGT visualization and analysis artifacts.</p>
<p><a class='pill' href='{html.escape(index_path.name)}'>this index</a></p></section>
<section class='grid'>{''.join(html_cards)}</section></main></body></html>
""",
        encoding="utf-8",
    )
    print(json.dumps({"schema": "toricgt.outputs_index.v1", "index_html": str(index_path), "cards": len(cards)}, indent=2))


if __name__ == "__main__":
    main()
