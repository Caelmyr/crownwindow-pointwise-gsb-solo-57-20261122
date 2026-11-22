"""Validation and canonical serialization for local observation documents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


class Problem(Exception):
    """A message suitable for displaying at the operation's point of use."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise Problem(f"{label} must be an object.")
    return value


def fields(value: dict, *allowed: str) -> None:
    extras = sorted(set(value) - set(allowed))
    if extras:
        raise Problem(f"Unexpected field: {extras[0]}.")


def text(value: Any, label: str, length: int, *, default: str | None = None) -> str:
    if value is None and default is not None:
        value = default
    if not isinstance(value, str):
        raise Problem(f"{label} must be text.")
    clean = value.strip()
    if (not clean and default is None) or len(clean) > length:
        raise Problem(f"{label} must have at most {length} characters.")
    if any(ord(c) < 32 and c not in "\n\t" for c in clean):
        raise Problem(f"{label} contains a control character.")
    return clean


def real(value: Any, label: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Problem(f"{label} must be a number.")
    if not math.isfinite(value) or not lo <= value <= hi:
        raise Problem(f"{label} must be between {lo:g} and {hi:g}.")
    return float(value)


def whole(value: Any, label: str, lo: int, hi: int) -> int:
    number = real(value, label, lo, hi)
    if number != int(number):
        raise Problem(f"{label} must be a whole number.")
    return int(number)


def boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise Problem(f"{label} must be true or false.")
    return value


def enum(value: Any, label: str, keys: tuple[str, ...]) -> str:
    if value not in keys:
        raise Problem(f"{label} must be one of {', '.join(keys)}.")
    return value


def items(data: Any, label: str, low: int, high: int) -> list:
    if not isinstance(data, list) or not low <= len(data) <= high:
        raise Problem(f"{label} requires {low} to {high} entries.")
    return data


def point(raw: Any, label: str = "Image point") -> tuple[float, float]:
    x, y = items(raw, label, 2, 2)
    return real(x, label + " x", 0, 1), real(y, label + " y", 0, 1)


def angle(raw: Any, label: str = "Direction") -> tuple[float, float]:
    b, h = items(raw, label, 2, 2)
    return real(b, label + " bearing", -360, 720) % 360, real(h, label + " elevation", 0, 90)


def timestamp(value: Any, label: str) -> str:
    value = text(value, label, 50)
    try:
        result = datetime.fromisoformat(value)
    except ValueError as error:
        raise Problem(f"{label} must be a date and time in ISO format.") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise Problem(f"{label} must include the time zone offset.")
    return result.astimezone(timezone.utc).isoformat(timespec="seconds")


def identifier(raw: Any, label: str = "Record") -> str:
    if not isinstance(raw, str) or not re.fullmatch("[a-f0-9]{32}", raw):
        raise Problem(f"{label} identifier was not found.", 404)
    return raw


def token(raw: Any, label: str = "Name") -> str:
    value = text(raw, label, 70)
    if not re.fullmatch("[a-zA-Z0-9][a-zA-Z0-9_.-]*", value) or ".." in value:
        raise Problem(f"{label} contains an unsupported character.")
    return value


def filename(raw: Any, fallback: str = "observation") -> str:
    value = re.sub(r"[^\w.-]", "-", Path(str(raw)).name, flags=re.ASCII)
    value = value.strip(".-_")[:95]
    return value or fallback


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def key_of(value: Any) -> str:
    return sha(canonical(value))


def number(value: Any, digits: int = 5) -> float | None:
    if value is None:
        return None
    number_value = float(value)
    return round(number_value, digits) if math.isfinite(number_value) else None


def paired(raw: Any, label: str) -> tuple[float, float]:
    start, stop = items(raw, label, 2, 2)
    first = real(start, label + " start", 0, 90)
    last = real(stop, label + " end", 0, 90)
    if last <= first:
        raise Problem(f"{label} must have a positive height.")
    return first, last


@dataclass(frozen=True)
class Station:
    name: str
    accession: str
    note: str
    synthetic: bool

    @classmethod
    def parse(cls, raw: Any) -> Station:
        data = mapping(raw, "Station")
        fields(data, "name", "accession", "note", "synthetic")
        return cls(
            text(data.get("name"), "Station name", 96),
            text(data.get("accession"), "Field reference", 70, default=""),
            text(data.get("note"), "Field note", 700, default=""),
            boolean(data.get("synthetic", False), "Synthetic flag"),
        )

    def public(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Capture:
    title: str
    recorded_at: str
    note: str
    exposure_role: str
    time_label: str

    @classmethod
    def parse(cls, raw: Any, fallback: str = "Overhead capture") -> Capture:
        data = mapping(raw, "Capture")
        fields(data, "title", "recorded_at", "note", "exposure_role", "time_label")
        return cls(
            text(data.get("title") or fallback, "Capture title", 96),
            timestamp(data.get("recorded_at", utc_now()), "Recorded at"),
            text(data.get("note"), "Capture note", 600, default=""),
            enum(data.get("exposure_role", "survey"), "Exposure role", ("survey", "reference", "comparison")),
            text(data.get("time_label"), "Time label", 70, default=""),
        )

    def public(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Footprint:
    """An azimuth sweep relative to a bearing, with an elevation interval."""

    bearing: float
    sweep: float
    bottom: float
    top: float

    @classmethod
    def all(cls) -> Footprint:
        return cls(0.0, 360.0, 0.0, 90.0)

    @classmethod
    def parse(cls, raw: Any) -> Footprint:
        if raw is None:
            return cls.all()
        data = mapping(raw, "Viewing sector")
        fields(data, "bearing", "sweep", "elevation")
        bearing = real(data.get("bearing", 0), "Start bearing", -360, 720) % 360
        sweep = real(data.get("sweep", 360), "Bearing sweep", .25, 360)
        bottom, top = paired(data.get("elevation", [0, 90]), "Elevation")
        return cls(bearing, sweep, bottom, top)

    def public(self) -> dict:
        return {"bearing": self.bearing, "sweep": self.sweep, "elevation": [self.bottom, self.top]}

    @property
    def steradians(self) -> float:
        top = math.sin(math.radians(self.top))
        bottom = math.sin(math.radians(self.bottom))
        return math.radians(self.sweep) * (top - bottom)


@dataclass(frozen=True)
class Line:
    start: tuple[float, float]
    end: tuple[float, float]

    @classmethod
    def parse(cls, raw: Any) -> Line:
        start, end = items(raw, "Line", 2, 2)
        a, b = point(start), point(end)
        if math.dist(a, b) < .003:
            raise Problem("Place the two endpoints farther apart.")
        return cls(a, b)

    def public(self) -> list[list[float]]:
        return [list(self.start), list(self.end)]


@dataclass(frozen=True)
class Palette:
    sky: tuple[int, int, int] = (150, 214, 235)
    blocked: tuple[int, int, int] = (38, 94, 73)
    unknown: tuple[int, int, int] = (232, 173, 91)
    missing: tuple[int, int, int] = (45, 54, 65)

    def public(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Mark:
    """Field label attached to a native image point or known direction."""

    title: str
    native: tuple[float, float] | None
    direction: tuple[float, float] | None
    role: str
    note: str

    @classmethod
    def parse(cls, raw: Any) -> Mark:
        data = mapping(raw, "Field mark")
        fields(data, "title", "native", "direction", "role", "note")
        native = None if data.get("native") is None else point(data["native"])
        direction = None if data.get("direction") is None else angle(data["direction"])
        if native is None and direction is None:
            raise Problem("A mark must point to the image or a direction.")
        return cls(
            text(data.get("title"), "Mark title", 80),
            native,
            direction,
            enum(data.get("role", "reference"), "Mark role", ("reference", "obstruction", "opening", "note")),
            text(data.get("note"), "Mark note", 350, default=""),
        )

    def public(self) -> dict:
        return asdict(self)
