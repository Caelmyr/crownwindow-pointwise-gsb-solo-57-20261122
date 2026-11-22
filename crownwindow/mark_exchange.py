"""Portable source-bound field notes for repeat visits."""

from __future__ import annotations

import csv
import io
import json

from .contracts import Mark, Problem, canonical, fields, identifier, items, mapping, sha, text
from .library import Library


NOTE_FORMAT = "crownwindow-field-marks-v1"
MAX_NOTE_BYTES = 300_000


def _usable_mark(record: dict) -> dict:
    return {field: record.get(field) for field in ("title", "native", "direction", "role", "note")}


def field_log(library: Library, photo: str, *, csv_only: bool = False) -> bytes:
    source = library.photo(photo)
    marks = library.marks(photo)
    if csv_only:
        return _csv_marks(marks)
    entries = [_usable_mark(record) for record in marks]
    document = {
        "format": NOTE_FORMAT,
        "observation_id": photo,
        "observation_name": source["capture"]["title"],
        "image_digest": source["image"]["sha256"],
        "lens_context": source["active_lens"],
        "marks": entries,
        "marks_sha256": sha(canonical(entries)),
    }
    return canonical(document)


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\r", "\t")):
        return "'" + value
    return value


def _csv_marks(entries: list[dict]) -> bytes:
    sink = io.StringIO()
    writer = csv.writer(sink)
    writer.writerow(["label", "role", "native_x", "native_y", "bearing", "elevation", "note", "saved_lens", "saved_at"])
    for item in entries:
        u, v = item["native"] or (None, None)
        bearing, elevation = item["direction"] or (None, None)
        writer.writerow([_cell(value) for value in (item["title"], item["role"], u, v, bearing, elevation,
                                                    item["note"], item["lens_id"], item["created_at"])])
    return sink.getvalue().encode("utf-8")


def _decode_log(raw: bytes) -> dict:
    if len(raw) > MAX_NOTE_BYTES:
        raise Problem("The note exchange file exceeds 300 kB.", 413)
    try:
        unpacked = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Problem("The notes file is not readable JSON.") from error
    doc = mapping(unpacked, "Field notes")
    fields(doc, "format", "observation_id", "observation_name", "image_digest", "lens_context", "marks", "marks_sha256")
    if doc.get("format") != NOTE_FORMAT:
        raise Problem("The notes file is not an exported CrownWindow field log.")
    identifier(doc.get("observation_id"), "Exported observation")
    text(doc.get("observation_name"), "Exported observation name", 96)
    if doc.get("lens_context"):
        identifier(doc["lens_context"], "Exported lens")
    raw_marks = items(doc.get("marks"), "Exported marks", 1, 220)
    parsed = [Mark.parse(record) for record in raw_marks]
    if sha(canonical(raw_marks)) != doc.get("marks_sha256"):
        raise Problem("The field notes differ from their recorded digest.", 409)
    return {**doc, "marks": parsed}


def import_log(library: Library, photo: str, revision: int, raw: bytes, *, commit: bool) -> dict:
    original = _decode_log(raw)
    target = library.photo(photo)
    if revision != target["revision"]:
        raise Problem("This observation has changed since the notes dialog opened.", 409)
    if target["image"]["sha256"] != original["image_digest"]:
        raise Problem("Exported native marks belong to different source pixels.", 409)
    present = {sha(canonical(_usable_mark(value))) for value in library.marks(photo)}
    new = []
    for mark in original["marks"]:
        key = sha(canonical(mark.public()))
        if key not in present:
            new.append(mark)
            present.add(key)
    attached = []
    if commit and new:
        attached = library.add_marks(photo, revision, new, deduplicate=True)
    return {
        "source_title": original["observation_name"],
        "target_title": target["capture"]["title"],
        "source_match": True,
        "original_lens": original["lens_context"],
        "current_lens": target["active_lens"],
        "same_lens_context": original["lens_context"] == target["active_lens"],
        "marks_found": len(original["marks"]),
        "new": [record.public() for record in new],
        "saved": attached,
        "already_present": len(original["marks"]) - len(new),
    }
