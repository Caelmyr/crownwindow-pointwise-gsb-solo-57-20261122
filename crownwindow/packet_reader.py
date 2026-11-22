"""A read-only viewer for verified CrownWindow portable evidence."""

from __future__ import annotations

import base64
import io
import json
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from .archive import Member, verify_packet
from .camera import bounded_size, encode_png
from .contracts import Problem, identifier, key_of, mapping, sha, text
from .visual import DARK, GOLD, MINT, MUTED, PALE, _draw_font


MAX_JSON = 2_600_000
MAX_FIGURE = 6_000_000
MAX_THUMBNAILS = 5


def _read_inspection(archive: zipfile.ZipFile) -> dict:
    entry = archive.getinfo("observation.json")
    if entry.file_size > MAX_JSON:
        raise Problem("The observation document exceeds the reading budget.", 413)
    try:
        documents = json.loads(archive.read(entry))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Problem("Observation details are not readable JSON.", 409) from error
    record = mapping(documents, "Observation document")
    return record


def _consistency(record: dict, receipt: dict):
    key = identifier(record.get("id"), "Retained observation")
    station = identifier(record.get("station"), "Station")
    title = text(record.get("name"), "Retained title", 100)
    tool = text(record.get("tool"), "Observation tool", 42)
    version = text(record.get("measurement_version"), "Measurement version", 60)
    request = mapping(record.get("order"), "Stored inspection request")
    views = record.get("sources")
    if not isinstance(views, list) or not 1 <= len(views) <= 4:
        raise Problem("The report did not identify its source photographs.", 409)
    check = key_of({"version": version, "station": station,
                    "tool": tool, "order": request, "sources": views})
    if check != record.get("fingerprint") or check != receipt.get("fingerprint") or key != receipt.get("source_id"):
        raise Problem("The report context does not agree with the retained fingerprint.", 409)
    return key, station, title, tool, version


def _photograph_manifest(record: dict, archive: zipfile.ZipFile, receipt: dict) -> list[dict]:
    candidates = [item["path"] for item in receipt["files"] if item["path"].startswith("originals/")]
    candidate_hash = {name: sha(archive.read(name)) for name in candidates}
    entries = []
    distinct_bytes = set()
    for item in record["sources"]:
        source = mapping(item, "Photograph")
        image = mapping(source.get("image"), "Photograph description")
        capture = mapping(source.get("capture"), "Photograph capture")
        key = identifier(source.get("id"), "Photograph")
        title = text(capture.get("title"), "Capture title", 96)
        digest = image.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise Problem("A source photograph has no usable digest.", 409)
        distinct_bytes.add(digest)
        matches = [name for name, value in candidate_hash.items() if value == digest]
        if receipt["originals_included"] and not matches:
            raise Problem(f"The packet cannot prove the source image for {title}.", 409)
        entries.append({"id": key, "title": title, "sha256": digest, "original_member": matches[0] if matches else None,
                        "lens": source.get("lens_title"), "size": [image.get("width"), image.get("height")]})
    if (set(candidate_hash.values()) - distinct_bytes) or (candidates and len(candidates) != len(entries)):
        raise Problem("The archive contains unreferenced source photographs.", 409)
    return entries


def _safe_figures(archive, record: dict, receipt: dict) -> list[dict]:
    labels = record.get("asset_labels", {})
    if not isinstance(labels, dict):
        labels = {}
    entries = (item for item in receipt["files"]
               if item["path"].startswith("results/") and item["path"].endswith(".png"))
    figures = []
    for item in entries:
        if item["size"] > MAX_FIGURE or len(figures) >= MAX_THUMBNAILS:
            continue
        name = item["path"].split("/")[-1]
        parsed = archive.read(item["path"])
        try:
            with Image.open(io.BytesIO(parsed)) as image:
                if image.format != "PNG" or image.width * image.height > 5_000_000:
                    continue
                thumb = image.convert("RGBA").resize(bounded_size(image.width, image.height, 780),
                                                     Image.Resampling.LANCZOS)
        except (OSError, ValueError, Image.DecompressionBombError):
            continue
        figures.append({"file": name, "label": str(labels.get(name) or name)[:115],
                        "data": base64.b64encode(encode_png(thumb)).decode("ascii")})
    return figures


def inspect_packet(raw: bytes) -> dict:
    receipt = verify_packet(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        details = _read_inspection(archive)
        key, station, name, tool, version = _consistency(details, receipt)
        sources = _photograph_manifest(details, archive, receipt)
        figures = _safe_figures(archive, details, receipt)
    rows = details.get("readings") or []
    return {
        "receipt": receipt,
        "station_id": station, "record_id": key, "name": name, "tool": tool, "version": version,
        "title": str(details.get("title") or tool)[:140],
        "conclusion": str(details.get("conclusion") or "")[:1700],
        "metrics": mapping(details.get("metrics", {}), "Overview metrics"),
        "readings": rows[:40] if isinstance(rows, list) else [],
        "figures": figures, "sources": sources,
    }


def _stamp(draw, caption, limit, position, color, size):
    value = str(caption)[:limit]
    draw.text(position, value, fill=color, font=_draw_font(size))


def evidence_receipt_image(doc: dict) -> bytes:
    scale = 41
    wide = 950
    header = 157
    columns = doc["sources"]
    evidence = doc["receipt"]["files"]
    height = header + 58 + 87 * len(columns) + scale * len(evidence)
    image = Image.new("RGB", (wide, height), DARK)
    draw = ImageDraw.Draw(image)
    _stamp(draw, "CrownWindow · received evidence", 70, (24, 18), GOLD, 16)
    _stamp(draw, doc["name"], 55, (24, 53), PALE, 27)
    _stamp(draw, doc["title"], 100, (24, 93), PALE, 14)
    source_flag = ("Original bytes verified" if doc["receipt"]["originals_included"]
                   else "Result files verified · originals omitted")
    _stamp(draw, source_flag, 90, (24, 120), MINT, 14)
    for i, source in enumerate(columns):
        y = header + 87 * i
        draw.rectangle((23, y, wide - 24, y + 77), fill=(36, 50, 54))
        _stamp(draw, source["title"], 76, (36, y + 10), PALE, 15)
        _stamp(draw, source["sha256"], 68, (36, y + 34), MUTED, 12)
        _stamp(draw, source["lens"] or "Lens description unavailable", 80, (36, y + 53), MUTED, 12)
    cover = header + 87 * len(columns)
    _stamp(draw, "Verified archive members", 85, (24, cover + 11), PALE, 18)
    for index, value in enumerate(evidence):
        top = cover + 50 + index * scale
        shade = (37, 50, 53) if index % 2 else (31, 42, 46)
        draw.rectangle((24, top, wide - 24, top + scale - 3), fill=shade)
        _stamp(draw, value["path"], 91, (33, top + 8), PALE, 14)
        _stamp(draw, f"{value['size']:,} bytes", 22, (wide - 170, top + 7), MUTED, 14)
    return encode_png(image)
