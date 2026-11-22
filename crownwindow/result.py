"""Report assets and evidence previews shared by the normal tools."""

from __future__ import annotations

import base64
import csv
from dataclasses import dataclass, field
import io

import numpy as np
from PIL import Image

from .camera import encode_png
from .contracts import Problem, number, token


@dataclass(frozen=True)
class Asset:
    name: str
    mime: str
    data: bytes
    label: str

    def inline(self) -> dict:
        return {"name": self.name, "mime": self.mime, "label": self.label,
                "data": base64.b64encode(self.data).decode("ascii")}


@dataclass
class Result:
    name: str
    title: str
    conclusion: str
    metrics: dict = field(default_factory=dict)
    readings: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    assets: list[Asset] = field(default_factory=list)
    probe_kind: str | None = None

    def png(self, name: str, label: str, value: np.ndarray | Image.Image) -> None:
        self._append(name + ".png", "image/png", encode_png(value), label)

    def csv(self, name: str, label: str, rows: list[dict], columns: list[str] | None = None) -> None:
        if columns is None:
            columns = list(dict.fromkeys(key for row in rows for key in row))
        raw = _to_csv(rows, columns)
        self._append(name + ".csv", "text/csv; charset=utf-8", raw, label)

    def _append(self, name: str, mime: str, data: bytes, label: str) -> None:
        token(name)
        if any(asset.name == name for asset in self.assets):
            raise Problem("A result tried to publish two files under one name.", 500)
        self.assets.append(Asset(name, mime, data, label))

    def outline(self) -> dict:
        return {
            "tool": self.name,
            "title": self.title,
            "conclusion": self.conclusion,
            "metrics": self.metrics,
            "readings": self.readings[:180],
            "row_count": len(self.readings),
            "warnings": self.warnings,
            "probe_kind": self.probe_kind,
            "asset_labels": {entry.name: entry.label for entry in self.assets},
        }


def _spreadsheet(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(_spreadsheet(entry)) for entry in value)
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\r", "\t"):
        return "'" + value
    return value


def _to_csv(rows: list[dict], cols: list[str]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(cols)
    for row in rows:
        writer.writerow([_spreadsheet(row.get(column)) for column in cols])
    return buf.getvalue().encode("utf-8")


def readable_bounds(low, high, scale: float = 100.) -> list[float | None]:
    return [number(n * scale) if n is not None else None for n in (low, high)]
