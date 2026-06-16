#!/usr/bin/env python3
"""Render every HTML page in a bundle to PNG screenshots.

This is a strict browser-rendering utility for ToricGT analysis bundles.  It
requires Playwright/Chromium and Pillow; if either backend is unavailable the
command fails instead of emitting placeholder screenshots.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import async_playwright


CSS = """
:root{color-scheme:dark;--bg:#030712;--panel:#07111f;--text:#e8fbff;--muted:#91a8b7;--cyan:#37e8ff;--border:rgba(55,232,255,.28)}
body{margin:0;background:radial-gradient(circle at top left,#092238 0,#030712 44rem);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,sans-serif}
main{max-width:1280px;margin:0 auto;padding:28px 22px 64px}
.hero,.card{background:linear-gradient(180deg,rgba(11,23,40,.95),rgba(5,13,25,.97));border:1px solid var(--border);border-radius:8px}
.hero{padding:22px;margin-bottom:16px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}.card{padding:12px}
h1{margin:0 0 8px;font-size:28px}h2{font-size:14px;overflow-wrap:anywhere}p{color:var(--muted)}
a{color:var(--cyan);text-decoration:none}a:hover{text-decoration:underline}
img{max-width:100%;border:1px solid rgba(55,232,255,.18);border-radius:6px;background:#07111f}
.meta{display:flex;gap:8px;flex-wrap:wrap}.pill{border:1px solid var(--border);border-radius:999px;padding:5px 9px;color:#dff9ff;background:rgba(55,232,255,.08);font-size:12px}
.err{color:#ff9bd8}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Directory containing HTML pages to render.")
    parser.add_argument("--output-dir", required=True, help="Directory where screenshots/index/contact sheet are written.")
    parser.add_argument("--exclude-dir-name", action="append", default=["html_screenshots"], help="Directory basename to exclude. May be repeated.")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--wait-ms", type=int, default=1200)
    parser.add_argument("--timeout-ms", type=int, default=45000)
    parser.add_argument(
        "--wait-until",
        choices=["commit", "domcontentloaded", "load", "networkidle"],
        default="networkidle",
        help="Playwright page.goto wait condition.",
    )
    parser.add_argument(
        "--full-page",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture full-page screenshots. Use --no-full-page for very large interactive reports.",
    )
    parser.add_argument(
        "--viewport-slices",
        type=int,
        default=0,
        help="When --no-full-page is used, also capture up to this many vertical viewport slices per page.",
    )
    parser.add_argument("--contact-cols", type=int, default=3)
    parser.add_argument(
        "--link-from-source-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a screenshot/contact-sheet link block to source-dir/index.html when present.",
    )
    parser.add_argument(
        "--interaction-audit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture additional screenshots after moving known ToricGT sliders/toggles and revealing detail panels.",
    )
    parser.add_argument(
        "--interaction-delay-ms",
        type=int,
        default=500,
        help="Delay after each interaction-audit state before taking its screenshot.",
    )
    parser.add_argument(
        "--assert-branching-report",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail when a ToricGT branching reasoning report loses required threshold, slider, or detail-panel DOM evidence.",
    )
    return parser.parse_args()


def safe_slug(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "__", rel).replace("/", "__")
    return slug.replace(".html", "") + ".png"


def html_files(source_dir: Path, excluded: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in source_dir.rglob("*.html"):
        rel_parts = set(path.relative_to(source_dir).parts[:-1])
        if rel_parts & excluded:
            continue
        files.append(path)
    return sorted(files)


async def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


async def _capture_one(
    page: Any,
    target: Path,
    *,
    output_dir: Path,
    timeout_ms: int,
    full_page: bool,
) -> dict[str, Any]:
    await page.screenshot(path=str(target), full_page=bool(full_page), timeout=int(timeout_ms))
    width, height = await _image_size(target)
    return {
        "screenshot": str(target.relative_to(output_dir)),
        "width": int(width),
        "height": int(height),
    }


async def _generic_interaction(page: Any, mode: str) -> bool:
    """Move standard ToricGT controls when a page has no custom audit hook."""

    return bool(
        await page.evaluate(
            """(mode) => {
              const setRange = (id, fraction) => {
                const el = document.getElementById(id);
                if (!el) return false;
                const min = Number(el.min || 0);
                const max = Number(el.max || 0);
                const value = Math.round(min + (max - min) * fraction);
                el.value = String(value);
                el.dispatchEvent(new Event('input', {bubbles:true}));
                return true;
              };
              const setCheck = (id, checked) => {
                const el = document.getElementById(id);
                if (!el) return false;
                el.checked = Boolean(checked);
                el.dispatchEvent(new Event('change', {bubbles:true}));
                return true;
              };
              let touched = false;
              if (mode === 'full_mid') {
                touched = setRange('radius_slider', 0.55) || touched;
                touched = setRange('reasoning_level_slider', 0.55) || touched;
                document.getElementById('trajectory_plot')?.scrollIntoView({block:'center'});
              } else if (mode === 'full_triangles') {
                touched = setRange('radius_slider', 1.0) || touched;
                touched = setRange('reasoning_level_slider', 1.0) || touched;
                touched = setCheck('full_triangle_toggle', true) || touched;
                document.getElementById('trajectory_plot')?.scrollIntoView({block:'center'});
              } else if (mode === 'step_detail') {
                touched = setRange('step_radius_slider', 1.0) || touched;
                touched = setRange('decoding_order_slider', 1.0) || touched;
                document.getElementById('step_plot')?.scrollIntoView({block:'center'});
              } else if (mode === 'step_triangles') {
                touched = setRange('step_radius_slider', 1.0) || touched;
                touched = setRange('decoding_order_slider', 1.0) || touched;
                touched = setCheck('step_triangle_toggle', true) || touched;
                document.getElementById('step_plot')?.scrollIntoView({block:'center'});
              } else if (mode === 'analogy_detail') {
                touched = setRange('analogy_radius_slider', 0.65) || touched;
                touched = setRange('analogy_reasoning_level_slider', 1.0) || touched;
                document.getElementById('analogy_plot')?.scrollIntoView({block:'center'});
              } else if (mode === 'analogy_triangles') {
                touched = setRange('analogy_radius_slider', 1.0) || touched;
                touched = setRange('analogy_reasoning_level_slider', 1.0) || touched;
                touched = setCheck('analogy_triangle_toggle', true) || touched;
                document.getElementById('analogy_plot')?.scrollIntoView({block:'center'});
              }
              return touched;
            }""",
            mode,
        )
    )


async def _run_interaction_audit(
    page: Any,
    *,
    base_target: Path,
    output_dir: Path,
    timeout_ms: int,
    full_page: bool,
    delay_ms: int,
) -> list[dict[str, Any]]:
    modes = [
        "full_mid",
        "full_triangles",
        "step_detail",
        "step_triangles",
        "analogy_detail",
        "analogy_triangles",
    ]
    has_hook = bool(await page.evaluate("typeof window.toricgtScreenshotAudit === 'function'"))
    has_controls = bool(
        await page.evaluate(
            "Boolean(document.querySelector('#radius_slider,#step_radius_slider,#analogy_radius_slider'))"
        )
    )
    if not has_hook and not has_controls:
        return []
    records: list[dict[str, Any]] = []
    for idx, mode in enumerate(modes):
        target = base_target.with_name(base_target.stem + f"__interaction_{idx:02d}_{mode}" + base_target.suffix)
        status = "ok"
        error = ""
        width = 0
        height = 0
        screenshot = ""
        try:
            touched = False
            if has_hook:
                touched = bool(await page.evaluate("(mode) => window.toricgtScreenshotAudit(mode)", mode))
            if not touched:
                touched = await _generic_interaction(page, mode)
            if not touched:
                continue
            await page.wait_for_timeout(max(50, int(delay_ms)))
            captured = await _capture_one(
                page,
                target,
                output_dir=output_dir,
                timeout_ms=int(timeout_ms),
                full_page=bool(full_page),
            )
            screenshot = str(captured["screenshot"])
            width = int(captured["width"])
            height = int(captured["height"])
        except Exception as exc:
            status = "error"
            error = repr(exc)
        records.append(
            {
                "label": mode,
                "screenshot": screenshot,
                "status": status,
                "error": error,
                "width": int(width),
                "height": int(height),
            }
        )
    return records


async def _assert_branching_report_dom(page: Any) -> list[dict[str, Any]]:
    """Assert the interactive branching report still exposes its audit contract."""

    is_branching = bool(
        await page.evaluate(
            "Boolean(document.getElementById('top_analogy_decision_status_badges') && document.getElementById('trajectory_state_caption'))"
        )
    )
    if not is_branching:
        return []
    has_hook = bool(await page.evaluate("typeof window.toricgtScreenshotAudit === 'function'"))
    if has_hook:
        await page.evaluate("() => window.toricgtScreenshotAudit('step_detail')")
        await page.wait_for_timeout(100)
        await page.evaluate("() => window.toricgtScreenshotAudit('analogy_detail')")
        await page.wait_for_timeout(100)
    checks = [
        (
            "analogy_status_matches_priority_rule",
            """(() => {
              const a = trajectory_simplex_payload?.analogy;
              if (!a?.decision_summary) return false;
              const expected = a.decision_summary.strong.passed ? 'strong_analogy'
                : (a.decision_summary.weak.passed ? 'weak_analogy'
                : (a.decision_summary.candidate.passed ? 'candidate_analogy' : 'no_analogy'));
              return a.analogy_status === expected;
            })()""",
        ),
        (
            "compact_edge_birth_payload",
            """(() => {
              const p = trajectory_simplex_payload;
              return p?.distance_storage === 'edge_births'
                && Array.isArray(p.edge_births)
                && p?.analogy?.source_simplex_tree?.edge_storage === 'compact_edge_births'
                && Array.isArray(p.analogy.source_simplex_tree.edge_births)
                && p?.analogy?.candidate_map?.map_image_storage === 'compact_arrays';
            })()""",
        ),
        (
            "top_analogy_status_badges",
            "Boolean(document.getElementById('top_analogy_decision_status_badges')?.textContent.includes('tier'))",
        ),
        (
            "top_threshold_table_step_gate",
            "Boolean(document.getElementById('top_analogy_threshold_table')?.textContent.includes('step simplex-map') && document.getElementById('top_analogy_threshold_table')?.textContent.includes('advisory'))",
        ),
        (
            "full_slider_caption",
            "Boolean(document.getElementById('trajectory_state_caption')?.textContent.includes('visible one-dimensional simplex edges'))",
        ),
        (
            "step_slider_caption",
            "Boolean(document.getElementById('step_state_caption')?.textContent.includes('visible tokens'))",
        ),
        (
            "analogy_slider_caption",
            "Boolean(document.getElementById('analogy_state_caption')?.textContent.includes('analogy'))",
        ),
        (
            "token_detail_panel",
            "Boolean(document.getElementById('token_detail_panel')?.textContent.includes('Token') && document.getElementById('token_detail_panel')?.textContent.includes('NLL'))",
        ),
        (
            "analogy_detail_panel",
            "Boolean(document.getElementById('analogy_detail_panel')?.textContent.includes('analogy vertex') && document.getElementById('analogy_detail_panel')?.textContent.includes('original embedding'))",
        ),
        (
            "compact_map_image_sample",
            "Boolean(document.getElementById('simplicial_map_validity_table')?.textContent.includes('Compact Map-Image Sample'))",
        ),
        (
            "map_confidence_summary",
            "Boolean(document.getElementById('simplicial_map_validity_table')?.textContent.includes('map confidence') && document.getElementById('analogy_map_summary')?.textContent.includes('vertex-distance quality'))",
        ),
        (
            "ph_feature_family_table",
            "Boolean(document.getElementById('ph_feature_family_table')?.textContent.includes('Vectorized') || document.getElementById('ph_feature_family_table')?.textContent.includes('landscape'))",
        ),
    ]
    records: list[dict[str, Any]] = []
    for label, expression in checks:
        try:
            passed = bool(await page.evaluate(expression))
            error = ""
        except Exception as exc:
            passed = False
            error = repr(exc)
        records.append({"label": label, "status": "ok" if passed else "error", "error": error})
    return records


async def capture_pages(
    files: list[Path],
    *,
    source_dir: Path,
    output_dir: Path,
    width: int,
    height: int,
    wait_ms: int,
    timeout_ms: int,
    wait_until: str,
    full_page: bool,
    viewport_slices: int,
    interaction_audit: bool,
    interaction_delay_ms: int,
    assert_branching_report: bool,
) -> list[dict[str, Any]]:
    screenshots = output_dir / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": int(width), "height": int(height)}, device_scale_factor=1)
        for path in files:
            target = screenshots / safe_slug(path, source_dir)
            rel = path.relative_to(source_dir).as_posix()
            status = "ok"
            error = ""
            shot_width = 0
            shot_height = 0
            slice_records: list[str] = []
            interaction_records: list[dict[str, Any]] = []
            assertion_records: list[dict[str, Any]] = []
            try:
                await page.goto(path.resolve().as_uri(), wait_until=str(wait_until), timeout=int(timeout_ms))
                await page.wait_for_timeout(int(wait_ms))
                base_target = target
                if not bool(full_page) and int(viewport_slices) > 0:
                    scroll_height = int(await page.evaluate("document.documentElement.scrollHeight || document.body.scrollHeight || 0"))
                    step = max(1, int(height * 0.88))
                    starts = list(range(0, max(scroll_height, height), step))[: int(viewport_slices)]
                    if not starts:
                        starts = [0]
                    for idx, scroll_y in enumerate(starts):
                        await page.evaluate("(y) => window.scrollTo(0, y)", int(scroll_y))
                        await page.wait_for_timeout(max(150, int(wait_ms // 4)))
                        slice_target = target.with_name(target.stem + f"__slice_{idx:02d}" + target.suffix)
                        await page.screenshot(path=str(slice_target), full_page=False, timeout=int(timeout_ms))
                        slice_records.append(str(slice_target.relative_to(output_dir)))
                    target = output_dir / slice_records[0]
                else:
                    await page.screenshot(path=str(target), full_page=bool(full_page), timeout=int(timeout_ms))
                with Image.open(target) as image:
                    shot_width, shot_height = image.size
                if bool(interaction_audit):
                    interaction_records = await _run_interaction_audit(
                        page,
                        base_target=base_target,
                        output_dir=output_dir,
                        timeout_ms=int(timeout_ms),
                        full_page=bool(full_page),
                        delay_ms=int(interaction_delay_ms),
                    )
                if bool(assert_branching_report):
                    assertion_records = await _assert_branching_report_dom(page)
                    failed_assertions = [
                        str(item.get("label", "assertion"))
                        for item in assertion_records
                        if item.get("status") != "ok"
                    ]
                    if failed_assertions:
                        raise AssertionError("branching report assertions failed: " + ", ".join(failed_assertions))
            except Exception as exc:
                status = "error"
                error = repr(exc)
            records.append(
                {
                    "html": rel,
                    "screenshot": str(target.relative_to(output_dir)) if target.exists() else "",
                    "status": status,
                    "error": error,
                    "width": int(shot_width),
                    "height": int(shot_height),
                    "slices": slice_records,
                    "interaction_screenshots": interaction_records,
                    "assertions": assertion_records,
                }
            )
        await browser.close()
    return records


def write_contact_sheet(records: list[dict[str, Any]], output_dir: Path, *, cols: int = 3) -> Path:
    ok: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") == "ok" and record.get("screenshot"):
            ok.append(
                {
                    "html": str(record.get("html", "")),
                    "screenshot": str(record.get("screenshot", "")),
                    "width": int(record.get("width", 0) or 0),
                    "height": int(record.get("height", 0) or 0),
                }
            )
        for interaction in record.get("interaction_screenshots", []) or []:
            if isinstance(interaction, dict) and interaction.get("status") == "ok" and interaction.get("screenshot"):
                ok.append(
                    {
                        "html": f"{record.get('html', '')} :: {interaction.get('label', 'interaction')}",
                        "screenshot": str(interaction.get("screenshot", "")),
                        "width": int(interaction.get("width", 0) or 0),
                        "height": int(interaction.get("height", 0) or 0),
                    }
                )
    thumb_w, thumb_h = 360, 230
    pad, label_h = 18, 52
    cols = max(1, int(cols))
    rows = max(1, (len(ok) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * (thumb_w + pad) + pad, rows * (thumb_h + label_h + pad) + pad), (3, 7, 18))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        small = ImageFont.truetype("DejaVuSans.ttf", 11)
    except Exception:
        font = small = None
    for idx, record in enumerate(ok):
        x = pad + (idx % cols) * (thumb_w + pad)
        y = pad + (idx // cols) * (thumb_h + label_h + pad)
        image = Image.open(output_dir / str(record["screenshot"])).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        frame = Image.new("RGB", (thumb_w, thumb_h), (7, 17, 31))
        frame.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
        sheet.paste(frame, (x, y))
        draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=(55, 232, 255), width=1)
        label = str(record["html"])
        if len(label) > 72:
            label = label[:69] + "..."
        draw.text((x, y + thumb_h + 8), label, fill=(232, 251, 255), font=font)
        draw.text((x, y + thumb_h + 28), f"{record['width']}x{record['height']}", fill=(145, 168, 183), font=small)
    contact = output_dir / "contact_sheet.png"
    sheet.save(contact)
    return contact


def write_index(records: list[dict[str, Any]], *, source_dir: Path, output_dir: Path, contact: Path) -> Path:
    cards: list[str] = []
    for record in records:
        cls = "err" if record.get("status") != "ok" else ""
        source = html.escape(source_dir.joinpath(str(record["html"])).resolve().as_uri())
        screenshot = html.escape(str(record.get("screenshot", "")))
        body = (
            f"<h2 class='{cls}'>{html.escape(str(record['html']))}</h2>"
            f"<div class='meta'><span class='pill'>{html.escape(str(record['status']))}</span>"
            f"<span class='pill'>{record.get('width', 0)} x {record.get('height', 0)}</span></div>"
        )
        if screenshot:
            body += f"<p><a href='{source}'>source html</a> · <a href='{screenshot}'>screenshot png</a></p><a href='{screenshot}'><img src='{screenshot}' loading='lazy'></a>"
        if record.get("error"):
            body += f"<p class='err'>{html.escape(str(record['error']))}</p>"
        interactions = record.get("interaction_screenshots", []) or []
        if interactions:
            items = []
            for item in interactions:
                if not isinstance(item, dict):
                    continue
                label = html.escape(str(item.get("label", "interaction")))
                status = html.escape(str(item.get("status", "")))
                shot = html.escape(str(item.get("screenshot", "")))
                if shot:
                    items.append(f"<li><a href='{shot}'>{label}</a> <span class='pill'>{status}</span></li>")
                else:
                    items.append(f"<li><span class='err'>{label}: {status}</span></li>")
            body += "<h2>Interaction States</h2><ul>" + "".join(items) + "</ul>"
        assertions = record.get("assertions", []) or []
        if assertions:
            items = []
            for item in assertions:
                if not isinstance(item, dict):
                    continue
                label = html.escape(str(item.get("label", "assertion")))
                status = html.escape(str(item.get("status", "")))
                cls = "err" if status != "ok" else ""
                error = html.escape(str(item.get("error", "")))
                items.append(f"<li class='{cls}'>{label}: <span class='pill'>{status}</span> {error}</li>")
            body += "<h2>DOM Assertions</h2><ul>" + "".join(items) + "</ul>"
        cards.append(f"<section class='card'>{body}</section>")
    ok_count = sum(1 for record in records if record.get("status") == "ok")
    index = output_dir / "index.html"
    index.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>ToricGT HTML Screenshots</title><style>{CSS}</style></head>
<body><main><section class="hero"><h1>ToricGT HTML Screenshot Audit</h1>
<p>Strict Playwright/Chromium screenshots for every selected HTML page under <code>{html.escape(str(source_dir))}</code>.</p>
<div class="meta"><span class="pill">{len(records)} pages</span><span class="pill">{ok_count} ok</span><span class="pill">{len(records)-ok_count} errors</span></div>
<p><a href="contact_sheet.png">contact sheet</a> · <a href="manifest.json">manifest JSON</a></p>
<img src="{html.escape(contact.name)}"></section><section class="grid">{''.join(cards)}</section></main></body></html>
""",
        encoding="utf-8",
    )
    return index


def link_from_source_index(source_dir: Path, *, index: Path, contact: Path, manifest: Path) -> bool:
    source_index = source_dir / "index.html"
    if not source_index.exists() or source_index.resolve() == index.resolve():
        return False
    rel_index = os.path.relpath(index, start=source_index.parent)
    rel_contact = os.path.relpath(contact, start=source_index.parent)
    rel_manifest = os.path.relpath(manifest, start=source_index.parent)
    marker_start = "<!-- toricgt-html-screenshot-link:start -->"
    marker_end = "<!-- toricgt-html-screenshot-link:end -->"
    block = f"""{marker_start}
<section style="border:1px solid rgba(55,232,255,.32);background:rgba(7,17,31,.92);border-radius:8px;padding:14px;margin:16px 0;color:#e8fbff;font-family:Inter,system-ui,sans-serif">
  <h2 style="margin:0 0 8px;font-size:18px;letter-spacing:0">Rendered HTML Screenshot Audit</h2>
  <p style="margin:0 0 8px;color:#9fb1c9">Strict browser screenshots and contact sheet for this analysis bundle.</p>
  <a style="color:#8dd6ff" href="{html.escape(rel_index)}">screenshot index</a>
  <span style="color:#607089"> · </span>
  <a style="color:#8dd6ff" href="{html.escape(rel_contact)}">contact sheet</a>
  <span style="color:#607089"> · </span>
  <a style="color:#8dd6ff" href="{html.escape(rel_manifest)}">screenshot manifest</a>
</section>
{marker_end}"""
    text = source_index.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(marker_start) + r".*?" + re.escape(marker_end), re.DOTALL)
    if marker_start in text:
        new_text = pattern.sub(block, text)
    elif "</main>" in text:
        new_text = text.replace("</main>", block + "\n</main>", 1)
    elif "</body>" in text:
        new_text = text.replace("</body>", block + "\n</body>", 1)
    else:
        new_text = text + "\n" + block + "\n"
    source_index.write_text(new_text, encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = html_files(source_dir, set(args.exclude_dir_name or []))
    if not files:
        raise RuntimeError(f"no HTML files found under {source_dir}")
    records = asyncio.run(
        capture_pages(
            files,
            source_dir=source_dir,
            output_dir=output_dir,
            width=int(args.width),
            height=int(args.height),
            wait_ms=int(args.wait_ms),
            timeout_ms=int(args.timeout_ms),
            wait_until=str(args.wait_until),
            full_page=bool(args.full_page),
            viewport_slices=int(args.viewport_slices),
            interaction_audit=bool(args.interaction_audit),
            interaction_delay_ms=int(args.interaction_delay_ms),
            assert_branching_report=bool(args.assert_branching_report),
        )
    )
    contact = write_contact_sheet(records, output_dir, cols=int(args.contact_cols))
    index = write_index(records, source_dir=source_dir, output_dir=output_dir, contact=contact)
    manifest_path = output_dir / "manifest.json"
    linked = False
    if bool(args.link_from_source_index):
        linked = link_from_source_index(source_dir, index=index, contact=contact, manifest=manifest_path)
    manifest = {
        "schema": "toricgt.html_screenshot_manifest.v1",
        "source_bundle": str(source_dir),
        "screenshots": records,
        "count": int(len(records)),
        "ok_count": int(sum(1 for record in records if record.get("status") == "ok")),
        "error_count": int(sum(1 for record in records if record.get("status") != "ok")),
        "contact_sheet": contact.name,
        "index_html": index.name,
        "source_index_linked": bool(linked),
        "interaction_audit": bool(args.interaction_audit),
        "interaction_count": int(
            sum(len(record.get("interaction_screenshots", []) or []) for record in records)
        ),
        "assert_branching_report": bool(args.assert_branching_report),
        "assertion_count": int(sum(len(record.get("assertions", []) or []) for record in records)),
        "assertion_error_count": int(
            sum(
                1
                for record in records
                for item in (record.get("assertions", []) or [])
                if isinstance(item, dict) and item.get("status") != "ok"
            )
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["error_count"] or manifest["assertion_error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
