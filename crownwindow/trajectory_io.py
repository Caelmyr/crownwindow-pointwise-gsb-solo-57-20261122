"""Import explicit direction landmarks from a field CSV."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from .contracts import Problem, canonical, number, sha
from .paths import PathMark


MAX_CSV = 384_000
ROWS_MAX = 96
DIRECT = {"time": "at", "datetime": "at", "at": "at", "utc": "at", "time_utc": "at", "timestamp": "at",
          "bearing": "bearing", "azimuth": "bearing", "azimuth_deg": "bearing", "bearing_deg": "bearing",
          "elevation": "elevation", "altitude": "elevation", "elevation_deg": "elevation", "altitude_deg": "elevation"}
IGNORE = ("comment", "label", "remark", "notes", "station")


def _decode(raw: bytes) -> str:
    if not raw or len(raw) > MAX_CSV:
        raise Problem("Choose a direction file smaller than 384 kB.", 413)
    try:
        data = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise Problem("Save the direction table as a UTF-8 CSV.") from error
    if "\x00" in data:
        raise Problem("Choose a text CSV rather than an application workbook.")
    return data


def _heading(definition: list[str]) -> dict:
    heading = {}
    for i, label in enumerate(definition):
        normalized = "_".join(label.strip().lower().replace("-", " ").split())
        if normalized in DIRECT:
            key = DIRECT[normalized]
            if key in heading:
                raise Problem(f"The direction table repeats the {key} heading.")
            heading[key] = i
        elif normalized not in IGNORE and normalized:
            raise Problem(f"Unrecognized direction column: {label}.")
    if sorted(heading) != ["at", "bearing", "elevation"]:
        raise Problem("The table needs time, bearing, and elevation headings.")
    return heading


def _location(raw: list[str], schema: dict, line: int):
    try:
        time = raw[schema["at"]].strip()
        bearing = float(raw[schema["bearing"]].strip())
        elevation = float(raw[schema["elevation"]].strip())
        return PathMark.parse({"at": time, "direction": [bearing, elevation]})
    except (IndexError, ValueError, Problem) as error:
        raise Problem(f"Direction row {line} needs an explicit time zone, numeric bearing and elevation.") from error


def read_path_csv(source: bytes, name: str) -> dict:
    contents = _decode(source)
    try:
        try:
            dialect = csv.Sniffer().sniff(contents[:8192], delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reads = csv.reader(io.StringIO(contents), dialect=dialect, strict=True)
        header = next(reads, None)
        if header is None:
            raise Problem("No header was found in the direction file.")
        schema = _heading(header)
        values = []
        for row in reads:
            if not row or not any(item.strip() for item in row):
                continue
            line = reads.line_num
            if len(row) != len(header):
                raise Problem(f"Direction row {line} does not match the header width.")
            angle = _location(row, schema, line)
            if values and values[-1].moment >= angle.moment:
                raise Problem(f"Direction row {line} is at or before the previous time.")
            values.append(angle)
            if len(values) > ROWS_MAX:
                raise Problem(f"A time path may contain up to {ROWS_MAX} field readings.", 413)
    except csv.Error as error:
        raise Problem("The direction table contains malformed CSV quoting.") from error
    if len(values) < 2:
        raise Problem("Add at least two timed line-of-sight observations.")
    start, end = values[0].moment, values[-1].moment
    if (end - start).total_seconds() > 49 * 3600:
        raise Problem("A direction CSV may cover at most 49 hours.")
    document = [value.public() for value in values]
    return {"filename": Path(name).name, "sha256": sha(source), "landmarks": document,
            "normalized_data_sha256": sha(canonical(document)),
            "first_at": start.isoformat(timespec="seconds"), "last_at": end.isoformat(timespec="seconds"),
            "span_hours": number((end - start).total_seconds() / 3600, 5), "count": len(values)}


def write_path_csv(points: list[dict]) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["timestamp", "bearing_deg", "elevation_deg"])
    for item in points:
        normalized = PathMark.parse(item)
        writer.writerow([normalized.moment.isoformat(timespec="seconds"), *normalized.direction])
    return out.getvalue().encode("utf-8")
