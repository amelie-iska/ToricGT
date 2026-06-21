#!/usr/bin/env python3
"""Render static PNG screenshots from a branching reasoning payload.

This is a companion to ``render_html_screenshots.py`` for very large
branching-reasoning reports whose fully interactive Plotly page is too heavy
for repeated browser screenshotting.  It reads the exact JSON payload emitted
by ``render_branching_reasoning_trajectory_report.py`` and draws deterministic
PNG audit screenshots without sampling away visible one-dimensional simplex
edges.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402


BG = "#030712"
PANEL = "#07111f"
GRID = "#18304a"
TEXT = "#e8fbff"
MUTED = "#9fb4c7"
CYAN = "#37e8ff"
MAGENTA = "#ff5be7"
YELLOW = "#ffd45a"
GREEN = "#7cff6b"
ORANGE = "#ffb47c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--radius-index", type=int, default=-1, help="Radius index for full trajectory screenshots. Default: midpoint.")
    parser.add_argument("--reasoning-level", type=int, default=-1, help="Reasoning level for full trajectory screenshots. Default: final level.")
    parser.add_argument("--step-id", type=int, default=-1, help="Reasoning step id for selected-step screenshot. Default: middle node.")
    parser.add_argument("--step-radius-index", type=int, default=-1, help="Radius index for selected-step screenshot. Default: final local radius.")
    parser.add_argument("--decode-order", type=int, default=-1, help="Decode order for selected-step screenshot. Default: final token.")
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1200)
    return parser.parse_args()


def _load_payload(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "nodes" not in data or "analogy" not in data:
        raise ValueError(f"{path} is not a branching reasoning payload")
    return data


def _node_xyz(nodes: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([node.get("pca", [0.0, 0.0, 0.0]) for node in nodes], dtype=np.float64)


def _node_nll(nodes: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(node.get("nll", 0.0)) for node in nodes], dtype=np.float64)


def _active_edge_pairs(edge_births: list[list[Any]], radius_idx: int, visible: set[int] | None = None) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for row in edge_births:
        if len(row) < 3:
            continue
        i, j, birth = int(row[0]), int(row[1]), int(row[2])
        if birth <= int(radius_idx) and (visible is None or (i in visible and j in visible)):
            pairs.append((i, j))
    return pairs


def _setup_3d(ax: Any, title: str) -> None:
    ax.set_title(title, color=TEXT, pad=18, fontsize=14)
    ax.set_facecolor(BG)
    ax.xaxis.set_pane_color((0.03, 0.07, 0.12, 1.0))
    ax.yaxis.set_pane_color((0.03, 0.07, 0.12, 1.0))
    ax.zaxis.set_pane_color((0.03, 0.07, 0.12, 1.0))
    ax.tick_params(colors=MUTED, labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = GRID
    ax.set_xlabel("PC1", color=MUTED)
    ax.set_ylabel("PC2", color=MUTED)
    ax.set_zlabel("PC3", color=MUTED)


def _save_fig(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def _draw_summary(payload: dict[str, Any], path: Path, *, width: int, height: int) -> None:
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    try:
        title = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
        head = ImageFont.truetype("DejaVuSans-Bold.ttf", 25)
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        mono = ImageFont.truetype("DejaVuSansMono.ttf", 19)
    except Exception:
        title = head = font = mono = None
    draw.rounded_rectangle([42, 42, width - 42, height - 42], radius=14, fill=PANEL, outline="#15566d", width=2)
    draw.text((82, 76), "Branching Graph-of-Thought Reasoning Trajectory", fill=TEXT, font=title)
    draw.text((82, 146), "Static screenshot audit from exact branching_reasoning_payload.json", fill=MUTED, font=font)
    analogy = payload["analogy"]
    scoreboard = analogy.get("decision_summary", {}).get("scoreboard", {})
    rows = [
        ("source mode", payload.get("source_mode", "unknown")),
        ("reasoning steps", len(payload.get("nodes", []))),
        ("DAG edges", len(payload.get("dag_edges", []))),
        ("radius levels", len(payload.get("radius_values", []))),
        ("analogy status", analogy.get("analogy_status")),
        ("analogy confidence", f"{float(analogy.get('analogy_confidence_score', analogy.get('analogy_confidence', 0.0))):.4f}"),
        (
            "full simplex-map valid fraction",
            f"{float(analogy.get('full_reasoning_trajectory_simplex_tree_map', {}).get('valid_fraction', 0.0)):.4f}",
        ),
        ("vectorized PH mean", f"{float(analogy.get('vectorized_persistence_cosine_mean', scoreboard.get('vectorized_ph_mean', 0.0))):.4f}"),
        ("PH gate", f"{float(scoreboard.get('ph_gate', 0.0)):.4f}"),
        ("step map mean", f"{float(analogy.get('step_simplicial_map_valid_fraction_mean', scoreboard.get('step_simplex_map_mean', 0.0))):.4f}"),
    ]
    y = 230
    draw.text((82, y), "Audit Metrics", fill=TEXT, font=head)
    y += 48
    for label, value in rows:
        draw.text((102, y), str(label), fill=MUTED, font=font)
        draw.text((760, y), str(value), fill=TEXT, font=mono)
        y += 42
    decision = analogy.get("decision_summary", {})
    y += 38
    draw.text((82, y), "Decision Rule", fill=TEXT, font=head)
    y += 46
    for tier in ("strong", "weak", "candidate"):
        row = decision.get(tier, {})
        result = "PASS" if row.get("passed") else "FAIL"
        color = GREEN if row.get("passed") else "#ff7aa8"
        draw.text((102, y), f"{tier}: {result}", fill=color, font=font)
        y += 34
    reason = str(analogy.get("analogy_reason", ""))
    for idx, line in enumerate([reason[i : i + 112] for i in range(0, len(reason), 112)][:5]):
        draw.text((102, y + 22 + idx * 30), line, fill=YELLOW, font=mono)
    image.save(path)


def _draw_full_trajectory(payload: dict[str, Any], path: Path, *, radius_idx: int, level: int) -> None:
    nodes = payload["nodes"]
    xyz = _node_xyz(nodes)
    nll = _node_nll(nodes)
    visible_ids = {int(node["id"]) for node in nodes if int(node.get("level", 0)) <= int(level)}
    pairs = _active_edge_pairs(payload.get("edge_births", []), radius_idx, visible_ids)
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    _setup_3d(ax, f"Full trajectory filtered complex · radius index {radius_idx} · reasoning level {level}")
    if pairs:
        segments = np.asarray([[xyz[i], xyz[j]] for i, j in pairs], dtype=np.float64)
        ax.add_collection3d(Line3DCollection(segments, colors=CYAN, linewidths=0.18, alpha=0.16))
    dag_edges = []
    for edge in payload.get("dag_edges", []):
        if int(edge[0]) in visible_ids and int(edge[1]) in visible_ids:
            dag_edges.append((int(edge[0]), int(edge[1]), str(edge[2]) if len(edge) > 2 else "edge"))
    if dag_edges:
        segments = np.asarray([[xyz[i], xyz[j]] for i, j, _ in dag_edges], dtype=np.float64)
        ax.add_collection3d(Line3DCollection(segments, colors=GREEN, linewidths=1.4, alpha=0.62))
    visible = sorted(visible_ids)
    scatter = ax.scatter(xyz[visible, 0], xyz[visible, 1], xyz[visible, 2], c=nll[visible], cmap="YlOrRd", s=20, depthshade=False)
    fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.02, label="node NLL")
    ax.text2D(
        0.02,
        0.02,
        f"visible vertices {len(visible)} · visible one-dimensional simplex edges {len(pairs)} · DAG arrows {len(dag_edges)}",
        transform=ax.transAxes,
        color=TEXT,
        fontsize=11,
    )
    _save_fig(fig, path)


def _selected_step(payload: dict[str, Any], step_id: int) -> dict[str, Any]:
    steps = payload.get("steps", [])
    if not steps:
        raise ValueError("payload has no per-step simplex data")
    if step_id < 0:
        return steps[len(steps) // 2]
    by_id = {int(step.get("step_index", step.get("id", idx))): step for idx, step in enumerate(steps)}
    return by_id.get(int(step_id), steps[len(steps) // 2])


def _draw_step(payload: dict[str, Any], path: Path, *, step_id: int, radius_idx: int, decode_order: int) -> None:
    step = _selected_step(payload, step_id)
    tokens = step.get("tokens", [])
    xyz = np.asarray([tok.get("pca", [0.0, 0.0, 0.0]) for tok in tokens], dtype=np.float64)
    nll = np.asarray([float(tok.get("nll", 0.0)) for tok in tokens], dtype=np.float64)
    if radius_idx < 0:
        radius_idx = max(0, len(step.get("radius_values", [])) - 1)
    if decode_order < 0:
        decode_order = max(0, len(tokens) - 1)
    visible = {idx for idx, tok in enumerate(tokens) if int(tok.get("decode_order", idx)) <= int(decode_order)}
    pairs = _active_edge_pairs(step.get("edge_births", []), radius_idx, visible)
    fig = plt.figure(figsize=(15, 9), facecolor=BG)
    ax = fig.add_subplot(121, projection="3d")
    _setup_3d(ax, f"Selected-step simplex tree · step {step.get('step_index', step.get('id'))} · radius index {radius_idx}")
    if pairs:
        ax.add_collection3d(Line3DCollection(np.asarray([[xyz[i], xyz[j]] for i, j in pairs]), colors=CYAN, linewidths=1.0, alpha=0.7))
    decode_pairs = [(idx, idx + 1) for idx in range(min(len(tokens) - 1, decode_order))]
    if decode_pairs:
        ax.add_collection3d(Line3DCollection(np.asarray([[xyz[i], xyz[j]] for i, j in decode_pairs]), colors=YELLOW, linewidths=1.0, linestyles="dotted", alpha=0.8))
    vis = sorted(visible)
    if vis:
        scatter = ax.scatter(xyz[vis, 0], xyz[vis, 1], xyz[vis, 2], c=nll[vis], cmap="YlOrRd", s=45, depthshade=False)
        fig.colorbar(scatter, ax=ax, shrink=0.55, pad=0.02, label="token NLL")
    for idx in vis[:24]:
        ax.text(xyz[idx, 0], xyz[idx, 1], xyz[idx, 2], str(idx), color=TEXT, fontsize=8)
    ax2 = fig.add_subplot(122)
    ax2.set_facecolor(BG)
    ax2.axis("off")
    ax2.text(0.0, 0.98, "Token metadata panel", color=TEXT, fontsize=18, fontweight="bold", va="top")
    lines = [
        f"step index: {step.get('step_index', step.get('id'))} · source level: {step.get('level')}",
        f"visible tokens: {len(vis)} · visible one-dimensional simplex edges: {len(pairs)}",
        f"decode order slider value: {decode_order}",
        "",
    ]
    for tok in tokens[: min(len(tokens), 16)]:
        lines.append(
            f"#{tok.get('decode_order', tok.get('id'))} {tok.get('text')} · type {tok.get('type')} · "
            f"NLL {float(tok.get('nll', 0.0)):.4f} · logprob {float(tok.get('logprob', 0.0)):.4f} · "
            f"entropy {float(tok.get('entropy', 0.0)):.4f} · rank {tok.get('rank')} · global {tok.get('global_token_id')}"
        )
    y = 0.90
    for line in lines[:22]:
        ax2.text(0.0, y, line, color=TEXT if line else MUTED, fontsize=10, va="top", family="monospace")
        y -= 0.045
    _save_fig(fig, path)


def _draw_analogy(payload: dict[str, Any], path: Path) -> None:
    analogy = payload["analogy"]
    source = analogy["source_simplex_tree"]
    memory = analogy["memory_simplex_tree"]
    sx = np.asarray([v.get("pca", [0.0, 0.0, 0.0]) for v in source.get("vertices", [])], dtype=np.float64)
    mx = np.asarray([v.get("pca", [0.0, 0.0, 0.0]) for v in memory.get("vertices", [])], dtype=np.float64)
    r_idx = max(0, len(payload.get("radius_values", [])) // 2)
    source_pairs = _active_edge_pairs(source.get("edge_births", []), r_idx)
    memory_pairs = _active_edge_pairs(memory.get("edge_births", []), r_idx)
    fig = plt.figure(figsize=(15, 9), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    _setup_3d(
        ax,
        f"Analogical memory map · {analogy.get('analogy_status')} · "
        f"confidence {float(analogy.get('analogy_confidence_score', analogy.get('analogy_confidence', 0.0))):.4f}",
    )
    if source_pairs:
        ax.add_collection3d(Line3DCollection(np.asarray([[sx[i], sx[j]] for i, j in source_pairs]), colors=CYAN, linewidths=0.35, alpha=0.35))
    if memory_pairs:
        ax.add_collection3d(Line3DCollection(np.asarray([[mx[i], mx[j]] for i, j in memory_pairs]), colors=MAGENTA, linewidths=0.35, alpha=0.35))
    n = min(len(sx), len(mx))
    if n:
        ax.scatter(sx[:, 0], sx[:, 1], sx[:, 2], c=ORANGE, s=18, depthshade=False, label="source simplex tree")
        ax.scatter(mx[:, 0], mx[:, 1], mx[:, 2], c="#ffd0a6", s=18, depthshade=False, label="retrieved memory simplex tree")
        map_rows = analogy.get("candidate_map", {}).get("vertex_images", [])
        for row in map_rows[:n]:
            i = int(row[0])
            j = int(row[1])
            if i < len(sx) and j < len(mx):
                ax.plot([sx[i, 0], mx[j, 0]], [sx[i, 1], mx[j, 1]], [sx[i, 2], mx[j, 2]], color=YELLOW, alpha=0.42, linewidth=0.6)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, loc="upper left")
    summary = analogy.get("candidate_map", {})
    split = analogy.get("display_separation", {})
    split_text = ""
    if isinstance(split, dict) and "x_axis_separation" in split:
        split_text = f" · display split {float(split.get('x_axis_separation', 0.0)):.2f} PC1 units"
    ax.text2D(
        0.02,
        0.02,
        "full map valid fraction "
        f"{float(analogy.get('full_reasoning_trajectory_simplex_tree_map', {}).get('valid_fraction', 0.0)):.4f} · "
        f"valid/collapsed fraction {float(summary.get('valid_fraction', 0.0)):.4f} · "
        f"vectorized PH mean {float(analogy.get('vectorized_persistence_cosine_mean', 0.0)):.4f}"
        f"{split_text}",
        transform=ax.transAxes,
        color=TEXT,
        fontsize=11,
    )
    _save_fig(fig, path)


def _draw_ph(payload: dict[str, Any], path: Path) -> None:
    analogy = payload["analogy"]
    family = analogy.get("vectorized_persistence_feature_family_summary", {})
    labels = []
    values = []
    family_means = family.get("family_means", {}) if isinstance(family, dict) else {}
    for name in family.get("families", []) if isinstance(family, dict) else []:
        if name in family_means:
            labels.append(str(name))
            values.append(float(family_means[name]))
    dimension_means = family.get("dimension_means", {}) if isinstance(family, dict) else {}
    for name, value in dimension_means.items():
        labels.append(str(name))
        values.append(float(value))
    if not labels:
        labels = ["landscape", "persistence_image", "silhouette", "entropy_vector"]
        values = [float(analogy.get("vectorized_persistence_cosine_mean", 0.0))] * len(labels)
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
    ax.set_facecolor(PANEL)
    y = np.arange(len(labels))
    ax.barh(y, values, color=[CYAN, MAGENTA, GREEN, YELLOW][: len(labels)])
    ax.set_yticks(y, labels=labels, color=TEXT)
    ax.set_xlim(0.0, 1.02)
    ax.tick_params(axis="x", colors=MUTED)
    ax.grid(axis="x", color=GRID, alpha=0.5)
    ax.set_title("Vectorized persistent-homology evidence near the analogical map", color=TEXT, fontsize=16)
    for idx, value in enumerate(values):
        ax.text(value + 0.01, idx, f"{value:.4f}", color=TEXT, va="center")
    _save_fig(fig, path)


def _write_contact(paths: list[Path], output_dir: Path) -> Path:
    images = [Image.open(path).convert("RGB") for path in paths]
    thumb_w, thumb_h = 440, 280
    pad, label_h = 18, 42
    cols = 2
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * (thumb_w + pad) + pad, rows * (thumb_h + label_h + pad) + pad), BG)
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font = None
    for idx, (image, path) in enumerate(zip(images, paths)):
        image.thumbnail((thumb_w, thumb_h))
        x = pad + (idx % cols) * (thumb_w + pad)
        y = pad + (idx // cols) * (thumb_h + label_h + pad)
        frame = Image.new("RGB", (thumb_w, thumb_h), PANEL)
        frame.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
        sheet.paste(frame, (x, y))
        draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=CYAN, width=1)
        draw.text((x, y + thumb_h + 8), path.name, fill=TEXT, font=font)
    contact = output_dir / "contact_sheet.png"
    sheet.save(contact)
    return contact


def _write_index(output_dir: Path, manifest: dict[str, Any]) -> Path:
    links = []
    for item in manifest["screenshots"]:
        path = html.escape(str(item["path"]))
        links.append(f"<li><a href='{path}'>{html.escape(str(item['label']))}</a></li>")
    index = output_dir / "index.html"
    index.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>Branching Payload Static Screenshots</title>
<style>body{{margin:0;background:{BG};color:{TEXT};font-family:Inter,system-ui,sans-serif}}main{{max-width:1100px;margin:0 auto;padding:28px}}a{{color:{CYAN}}}.card{{background:{PANEL};border:1px solid #15566d;border-radius:8px;padding:18px;margin:14px 0}}img{{max-width:100%;border:1px solid #15566d;border-radius:6px}}</style></head>
<body><main><section class="card"><h1>Branching Payload Static Screenshot Audit</h1><p>Generated directly from the exact branching payload for large reports.</p><p><a href="manifest.json">manifest JSON</a> · <a href="contact_sheet.png">contact sheet</a></p><img src="contact_sheet.png"></section><section class="card"><h2>Screenshots</h2><ul>{''.join(links)}</ul></section></main></body></html>""",
        encoding="utf-8",
    )
    return index


def main() -> None:
    args = parse_args()
    payload_path = Path(args.payload_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_payload(payload_path)
    radius_values = payload.get("radius_values", [])
    radius_idx = int(args.radius_index)
    if radius_idx < 0:
        radius_idx = max(0, len(radius_values) // 2)
    radius_idx = min(radius_idx, max(0, len(radius_values) - 1))
    max_level = max(int(node.get("level", 0)) for node in payload.get("nodes", [{"level": 0}]))
    level = int(args.reasoning_level)
    if level < 0:
        level = max_level
    step = _selected_step(payload, int(args.step_id))
    step_id = int(step.get("step_index", step.get("id", 0)))
    step_radius_idx = int(args.step_radius_index)
    if step_radius_idx < 0:
        step_radius_idx = max(0, len(step.get("radius_values", [])) - 1)
    decode_order = int(args.decode_order)
    if decode_order < 0:
        decode_order = max(0, len(step.get("tokens", [])) - 1)

    screenshots = [
        ("summary", output_dir / "summary.png"),
        ("full trajectory filtered complex", output_dir / "full_trajectory_filtered_complex.png"),
        ("selected step simplex tree and token metadata", output_dir / "selected_step_simplex_tree_tokens.png"),
        ("analogical memory simplex-tree map", output_dir / "analogical_memory_simplex_tree_map.png"),
        ("vectorized persistent-homology features", output_dir / "vectorized_ph_features.png"),
    ]
    _draw_summary(payload, screenshots[0][1], width=int(args.width), height=int(args.height))
    _draw_full_trajectory(payload, screenshots[1][1], radius_idx=radius_idx, level=level)
    _draw_step(payload, screenshots[2][1], step_id=step_id, radius_idx=step_radius_idx, decode_order=decode_order)
    _draw_analogy(payload, screenshots[3][1])
    _draw_ph(payload, screenshots[4][1])
    contact = _write_contact([path for _label, path in screenshots], output_dir)
    manifest = {
        "schema": "toricgt.branching_payload_static_screenshots.v1",
        "payload_json": str(payload_path),
        "nodes": len(payload.get("nodes", [])),
        "dag_edges": len(payload.get("dag_edges", [])),
        "radius_index": int(radius_idx),
        "reasoning_level": int(level),
        "selected_step_id": int(step_id),
        "step_radius_index": int(step_radius_idx),
        "decode_order": int(decode_order),
        "contact_sheet": contact.name,
        "screenshots": [{"label": label, "path": path.name} for label, path in screenshots],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    index = _write_index(output_dir, manifest)
    manifest["index_html"] = index.name
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
