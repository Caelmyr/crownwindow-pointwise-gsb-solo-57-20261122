"""Portable, digest-verified evidence with an offline reading sheet."""

from __future__ import annotations

from dataclasses import dataclass
import html
import io
import json
from pathlib import PurePosixPath
import re
import zipfile

from .contracts import Problem, canonical, filename, sha, token
from .library import Library


FORMAT = "crownwindow-evidence-v1"
MAX_PACKET_FILES = 36
MAX_PACKET_RAW = 110 * 1024 * 1024
MAX_PACKET_COMPRESSED = 65 * 1024 * 1024
MAX_SHEET_ROWS = 35
FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,123}$")


@dataclass(frozen=True)
class Member:
    path: str
    data: bytes
    mime: str
    title: str

    def metadata(self):
        return {"path": self.path, "size": len(self.data), "sha256": sha(self.data),
                "mime": self.mime, "title": self.title}


def _media(library: Library, record: dict, include_originals: bool) -> list[Member]:
    assets = []
    for candidate in record["files"]:
        name = candidate["name"]
        typed, raw = library.read_asset(record["id"], name)
        label = record.get("asset_labels", {}).get(name, name)
        assets.append(Member("results/" + token(name), raw, typed["mime"], label))
    if include_originals:
        for index, source in enumerate(record.get("sources", []), 1):
            data = library.source_blob(source)
            name = filename(source["image"]["file"], "source")
            mime = source["image"]["mime"]
            assets.append(Member(f"originals/{index:02d}-{name}", data, mime, source["capture"]["title"]))
    return assets


def _snapshot(record: dict) -> dict:
    keep = ("id", "station", "name", "created_at", "tool", "fingerprint", "title", "conclusion",
            "metrics", "readings", "row_count", "warnings", "order", "sources", "reviews", "measurement_version", "asset_labels")
    return {key: record[key] for key in keep if key in record}


def build_packet(library: Library, record: dict, include_originals: bool) -> tuple[str, bytes]:
    media = _media(library, record, include_originals)
    if len(media) > MAX_PACKET_FILES - 3:
        raise Problem("The report has too many separate assets for an evidence packet.", 413)
    raw = sum(len(item.data) for item in media)
    if raw > MAX_PACKET_RAW - 1_000_000:
        raise Problem("Export without the originals to keep the evidence within the portable packet budget.", 413)
    catalog = _snapshot(record)
    sheet = render_sheet(record, inline=False, assets=media)
    documents = (
        Member("reading-sheet.html", sheet.encode("utf-8"), "text/html; charset=utf-8", "Offline reading sheet"),
        Member("observation.json", canonical(catalog), "application/json", "Observation details"),
    )
    roster = [item.metadata() for item in (*media, *documents)]
    manifest = {"format": FORMAT, "source_id": record["id"], "fingerprint": record["fingerprint"],
                "originals_included": include_originals, "files": roster}
    package = io.BytesIO()
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as handle:
        handle.writestr("manifest.json", canonical(manifest))
        for item in (*documents, *media):
            handle.writestr(item.path, item.data)
    compressed = package.getvalue()
    if len(compressed) > MAX_PACKET_COMPRESSED:
        raise Problem("The portable packet would exceed 65 MiB. Export the results without originals.", 413)
    return filename(record["name"], "crownwindow") + ".zip", compressed


def _escape(value) -> str:
    if isinstance(value, dict):
        return html.escape(json.dumps(value, ensure_ascii=False), quote=True)
    if isinstance(value, (tuple, list)):
        return ", ".join(_escape(x) for x in value)
    return html.escape(str(value if value is not None else "—"), quote=True)


def _figures(record: dict, inline: bool, assets=()) -> str:
    pictures = []
    if inline:
        for item in record.get("files", []):
            if item["mime"] != "image/png":
                continue
            label = record.get("asset_labels", {}).get(item["name"], item["name"])
            pictures.append((item["url"], label))
    else:
        pictures = [(asset.path, asset.title) for asset in assets
                    if asset.mime == "image/png" and asset.path.startswith("results/")]
    return "".join(f"<figure><img src='{_escape(url)}' alt='{_escape(label)}'><figcaption>{_escape(label)}</figcaption></figure>"
                   for url, label in pictures[:9])


def _headword(key: str) -> str:
    return key.replace("_", " ").replace("-", " ").title()


def _short_table(rows: list[dict], maximum: int) -> str:
    if not rows:
        return "<p>There are no entries here.</p>"
    selected = list(dict.fromkeys(k for row in rows[:maximum] for k in row))[:12]
    headers = "".join(f"<th>{_escape(_headword(k))}</th>" for k in selected)
    body = []
    for row in rows[:maximum]:
        fragments = "".join(f"<td>{_escape(row.get(label))}</td>" for label in selected)
        body.append(f"<tr>{fragments}</tr>")
    return f"<div class='scroller'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _summaries(metrics: dict) -> str:
    blocks = []
    for key, value in metrics.items():
        if isinstance(value, (dict, list)) and len(value) > 4:
            continue
        label = _escape(_headword(key))
        rendered = _escape(value)
        blocks.append(f"<div><dt>{label}</dt><dd>{rendered}</dd></div>")
    return f"<dl>{''.join(blocks)}</dl>"


def _sources(record: dict, inline: bool, assets: list[Member]) -> str:
    links = iter([value.path for value in assets if value.path.startswith("originals/")])
    rows = []
    for source in record.get("sources", []):
        name = source["capture"]["title"]
        image = source["image"]
        link = source["original"] if inline else next(links, "")
        file_target = f"<a href='{_escape(link)}'>{_escape(image['file'])}</a>" if link else _escape(image["file"])
        size = _escape(f"{image['width']} × {image['height']} px")
        lens = _escape(source["lens_title"])
        rows.append(f"<li><b>{_escape(name)}</b> · {size} · {lens}<br>{file_target}<br>"
                    f"<small>{_escape(image['sha256'])}</small></li>")
    return f"<ol>{''.join(rows)}</ol>"


def render_sheet(record: dict, *, inline: bool = True, assets: list[Member] | None = None) -> str:
    name = _escape(record["name"])
    title = _escape(record.get("title", record["tool"]))
    stamp = _escape(record["created_at"])
    summary = _escape(record.get("conclusion", ""))
    figures = _figures(record, inline, assets or ())
    readings = _short_table(record.get("readings", []), MAX_SHEET_ROWS)
    sources = _sources(record, inline, assets or [])
    metrics = _summaries(record.get("metrics", {}))
    reviews = _short_table(record.get("reviews", []), 20)
    warnings = "".join(f"<li>{_escape(warning)}</li>" for warning in record.get("warnings", []))
    navigation = f"<p><a href='/'>Return to CrownWindow</a></p>" if inline else "<p>Offline copy of a retained observation.</p>"
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{name}</title>
<style>
body{{font:16px/1.55 system-ui,sans-serif;color:#1b3029;background:#faf8ef;padding:28px;max-width:1150px;margin:auto}}
header{{border-bottom:2px solid #125744;padding:14px 0 24px}} h1{{font:normal 43px Georgia,serif;margin:0}} h2{{font-size:21px;page-break-after:avoid}}
.sheet{{display:grid;grid-template-columns:repeat(auto-fit,minmax(285px,1fr));gap:16px}}
figure{{margin:0;break-inside:avoid}} img{{max-width:100%;height:auto;border-radius:4px}} figcaption{{color:#315446}}
dt{{font-weight:700}} dd{{margin:0}} dl{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:15px}}
.scroller{{overflow:auto}} table{{border-collapse:collapse;font-size:12px;width:100%}} th,td{{border-bottom:1px solid #b6c2b8;padding:5px;text-align:left}}
small{{overflow-wrap:anywhere}} ol li{{margin:10px 0}} @media print{{body{{padding:3mm;background:white}} header{{page-break-after:avoid}}}}
</style>
<header><p>CrownWindow / retained observation · {stamp}</p><h1>{name}</h1><p>{title}</p>{navigation}</header>
<main><p>{summary}</p><h2>Observed basis</h2><ul>{warnings}</ul>{metrics}<div class="sheet">{figures}</div>
<h2>Readings</h2>{readings}<h2>Original observations</h2>{sources}<h2>Review history</h2>{reviews}</main></html>"""


def _member_path(path: str) -> bool:
    if not isinstance(path, str):
        return False
    bits = PurePosixPath(path).parts
    return (len(bits) == 1 and path in ("manifest.json", "reading-sheet.html", "observation.json")) or (
        len(bits) == 2 and bits[0] in ("results", "originals") and bool(FILE_NAME.fullmatch(bits[1]))
        and path == "/".join(bits) and ".." not in bits[1])


def _read_manifest(archive: zipfile.ZipFile, names: set[str]) -> dict:
    if "manifest.json" not in names:
        raise Problem("This ZIP has no CrownWindow manifest.")
    block = archive.getinfo("manifest.json")
    if block.file_size > 250_000:
        raise Problem("The manifest is larger than expected.", 413)
    try:
        manifest = json.loads(archive.read(block))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Problem("The report manifest is not readable JSON.") from error
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise Problem("Unrecognized CrownWindow manifest format.")
    return manifest


def _validate_zip_headers(archive: zipfile.ZipFile) -> set[str]:
    parts = archive.infolist()
    if not 3 <= len(parts) <= MAX_PACKET_FILES:
        raise Problem("The archive has an unexpected number of entries.")
    names = [item.filename for item in parts]
    if len(set(names)) != len(names):
        raise Problem("The archive contains duplicate entry names.")
    if any(not _member_path(name) for name in names):
        raise Problem("The archive contains an unexpected file path.")
    expanded = sum(item.file_size for item in parts)
    if expanded > MAX_PACKET_RAW or any(item.flag_bits & 1 for item in parts):
        raise Problem("Encrypted or oversized observation packets are not accepted.", 413)
    return set(names)


def verify_packet(raw: bytes) -> dict:
    if not raw or len(raw) > MAX_PACKET_COMPRESSED:
        raise Problem("Choose an evidence ZIP smaller than 65 MiB.", 413)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = _validate_zip_headers(archive)
            manifest = _read_manifest(archive, names)
            expected = manifest.get("files")
            if not isinstance(expected, list) or len(expected) != len(names) - 1:
                raise Problem("The manifest does not account for every evidence file.")
            seen = set()
            output = []
            for entry in expected:
                if not isinstance(entry, dict):
                    raise Problem("An evidence entry is malformed.")
                name = entry.get("path")
                if not _member_path(name) or name in seen or name == "manifest.json" or name not in names:
                    raise Problem("The manifest names an invalid or repeated file.")
                seen.add(name)
                value = archive.read(name)
                if len(value) != entry.get("size") or sha(value) != entry.get("sha256"):
                    raise Problem(f"Evidence failed its byte check: {name}.", 409)
                output.append({"path": name, "size": len(value), "verified": True})
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError) as error:
        raise Problem("The evidence ZIP could not be read.") from error
    if not {"reading-sheet.html", "observation.json"}.issubset(seen):
        raise Problem("The reading sheet or structured observation is missing.")
    originals = any(row["path"].startswith("originals/") for row in output)
    if originals != manifest.get("originals_included"):
        raise Problem("The inclusion of original images does not match the manifest.", 409)
    return {"verified": True, "source_id": manifest.get("source_id"), "fingerprint": manifest.get("fingerprint"),
            "originals_included": originals, "files": output,
            "note": "The archive agrees internally with its manifest. This does not authenticate its author."}
