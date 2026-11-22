"""Station field folios tie together retained observations and today's originals."""

from __future__ import annotations

import csv
import html
import io

from PIL import Image, ImageDraw

from .camera import Camera, encode_png
from .contracts import Problem, filename, identifier, items, text
from .library import Library
from .visual import DARK, GOLD, MINT, MUTED, PALE, PANEL, _draw_font, photo_canvas


def _status(report: dict) -> str:
    reviews = report.get("reviews", [])
    if reviews:
        return reviews[-1]["status"]
    return "awaiting_review"


def _overlap(sources: list[dict]) -> list[str]:
    return list(dict.fromkeys(source["id"] for source in sources))


def _records(library: Library, station: str, specified: list | None) -> tuple[dict, list]:
    view = library.station_details(station)
    ids = [row["id"] for row in view["reports"][:16]] if specified is None else [
        identifier(raw, "Retained report") for raw in items(specified, "Selected retained reports", 1, 16)]
    if len(ids) != len(set(ids)):
        raise Problem("Select each retained report only once.")
    if not ids:
        raise Problem("Retain some inspection results before assembling a field folio.", 409)
    reports = [library.retained(key) for key in ids]
    if any(row["station"] != station for row in reports):
        raise Problem("All folio inspections must belong to the current station.", 409)
    return view, reports


def _outline_report(report: dict):
    return {
        "id": report["id"], "name": report["name"], "tool": report["tool"], "time": report["created_at"],
        "source_id": _overlap(report.get("sources", [])), "review_status": _status(report),
        "fingerprint": report["fingerprint"],
        "sheet": f"/api/retained/{report['id']}/sheet",
    }


def _timeline_report(report: dict, photo_rows: dict[str, dict]) -> list[dict]:
    records = []
    for source in report["sources"]:
        original = photo_rows.get(source["id"])
        records.append({
            "report": report["name"],
            "report_id": report["id"],
            "tool": report["tool"],
            "review_status": _status(report),
            "capture": source["capture"]["title"],
            "capture_at": source["capture"]["recorded_at"],
            "source": source["id"],
            "image_digest": source["image"]["sha256"],
            "lens_used": source["lens_title"],
            "active_lens_status": None if not original else original["active_lens"] == source.get("lens_id"),
        })
    return records


def make_folio(library: Library, key: str, selected: list | None, title: str):
    view, reports = _records(library, key, selected)
    active = {photo["id"]: photo for photo in view["photos"]}
    index_rows = []
    outlines = []
    source_outlines = {}
    for report in reports:
        outlines.append(_outline_report(report))
        index_rows.extend(_timeline_report(report, active))
        for instrument in report["sources"]:
            key = instrument["id"]
            source_outlines.setdefault(key, {"title": instrument["capture"]["title"],
                                              "digest": instrument["image"]["sha256"],
                                              "original": instrument["original"], "used": 0,
                                              "first_seen": instrument["capture"]["recorded_at"]})
            source_outlines[key]["used"] += 1
    mark = text(title, "Folio title", 80, default="") or view["station"]["name"]
    return {
        "title": mark, "station": view["station"], "reports": outlines,
        "sources": list(source_outlines.values()), "readings": index_rows,
        "report_count": len(reports), "source_count": len(source_outlines),
        "needs_review": sum(record["review_status"] != "confirmed" for record in outlines),
        "lens_versions_to_recheck": len({(row["source"], row["lens_used"]) for row in index_rows if row["active_lens_status"] is False}),
    }, reports


def _font_text(draw, position, value, color, size, *, limit=95):
    draw.text(position, str(value)[:limit], fill=color, font=_draw_font(size))


def contact_sheet(library: Library, body: dict, reports: list[dict]) -> bytes:
    width = 1100
    band = 156
    page = Image.new("RGB", (width, len(reports) * band + 118), DARK)
    sketch = ImageDraw.Draw(page)
    _font_text(sketch, (26, 17), body["title"], PALE, 28)
    _font_text(sketch, (28, 59), "Field folio  ·  " + body["station"]["accession"] + "  /  " +
               str(body["report_count"]) + " retained inspections", MUTED, 16)
    for i, report in enumerate(reports):
        top = 104 + band * i
        sketch.rectangle((26, top, width - 26, top + band - 10), fill=PANEL)
        _font_text(sketch, (42, top + 12), report["name"], PALE, 19, limit=62)
        _font_text(sketch, (44, top + 45), report.get("title") or report["tool"], MUTED, 13)
        current = _status(report)
        _font_text(sketch, (44, top + 69), current.replace("_", " "), MINT if current == "confirmed" else GOLD, 13)
        _font_text(sketch, (44, top + 96), report["created_at"], MUTED, 13)
        for j, source in enumerate(report["sources"][:4]):
            raw = library.source_blob(source)
            camera = Camera.read(raw, source["image"]["file"])
            thumbnail = photo_canvas(camera, edge=117)
            column = width - 162 * (j + 1)
            page.paste(thumbnail, (column, top + 11))
            _font_text(sketch, (column, top + 131), source["capture"]["title"], MUTED, 10, limit=20)
    return encode_png(page)


def _safe_spreadsheet(raw):
    if isinstance(raw, str) and raw.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + raw
    return raw


def folio_csv(document: dict) -> bytes:
    columns = ("report", "report_id", "tool", "review_status", "capture", "capture_at", "source",
               "image_digest", "lens_used", "active_lens_status")
    buf = io.StringIO()
    output = csv.writer(buf)
    output.writerow(columns)
    for row in document["readings"]:
        output.writerow([_safe_spreadsheet(row.get(key)) for key in columns])
    return buf.getvalue().encode("utf-8")


def render_folio(doc: dict) -> str:
    el = html.escape
    rows = []
    for value in doc["reports"]:
        file = el(value["sheet"], quote=True)
        label = el(value["name"])
        form = el(value["tool"].replace("-", " "))
        status = el(value["review_status"].replace("_", " "))
        day = el(value["time"])
        rows.append(f"<tr><td><a href='{file}'>{label}</a></td><td>{form}</td><td>{status}</td><td>{day}</td></tr>")
    sources = []
    for value in doc["sources"]:
        url = el(value["original"], quote=True)
        title = el(value["title"])
        count = int(value["used"])
        sources.append(f"<li><a href='{url}'>{title}</a> · {count} retained inspections<br><small>{el(value['digest'])}</small></li>")
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{el(doc['title'])} | Field folio</title><style>
body{{font:15px/1.65 system-ui;background:#f4f2e6;color:#20372d;max-width:1100px;padding:20px;margin:auto}}
header{{border-bottom:2px solid #266a56}} h1{{font:38px Georgia,serif}} a{{color:#196859}} small{{overflow-wrap:anywhere}}
table{{width:100%;border-collapse:collapse}} th,td{{text-align:left;border-bottom:1px solid #bccaad;padding:7px}}
</style><header><p>CrownWindow / field folio</p><h1>{el(doc['title'])}</h1><p>{el(doc['station']['accession'])} ·
{doc['report_count']} retained inspections and {doc['source_count']} source photographs</p></header>
<main><h2>Inspection register</h2><table><thead><tr><th>Inspection</th><th>Tool</th><th>Review</th><th>Retained at</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table><h2>Original camera photographs</h2><ul>{''.join(sources)}</ul>
<p>Lens groups to recheck against the current version: {doc['lens_versions_to_recheck']}.</p></main></html>"""


def folio_name(document: dict, extension: str):
    return filename(document["title"], "field-folio") + extension
