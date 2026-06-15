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
    parser.add_argument("--contact-cols", type=int, default=3)
    parser.add_argument(
        "--link-from-source-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a screenshot/contact-sheet link block to source-dir/index.html when present.",
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


async def capture_pages(
    files: list[Path],
    *,
    source_dir: Path,
    output_dir: Path,
    width: int,
    height: int,
    wait_ms: int,
    timeout_ms: int,
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
            try:
                await page.goto(path.resolve().as_uri(), wait_until="networkidle", timeout=int(timeout_ms))
                await page.wait_for_timeout(int(wait_ms))
                await page.screenshot(path=str(target), full_page=True, timeout=int(timeout_ms))
                with Image.open(target) as image:
                    shot_width, shot_height = image.size
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
                }
            )
        await browser.close()
    return records


def write_contact_sheet(records: list[dict[str, Any]], output_dir: Path, *, cols: int = 3) -> Path:
    ok = [record for record in records if record.get("status") == "ok" and record.get("screenshot")]
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
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
