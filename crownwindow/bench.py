"""Resolve selected photographs into an immutable calculation context."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from threading import BoundedSemaphore
from typing import Any

import numpy as np

from .camera import Camera
from .contracts import Footprint, Problem, boolean, fields, identifier, items, key_of, mapping, text, whole
from .library import Library
from .projection import Grid, Lens
from .sky import Classifier, Controls, DirectionSamples, sample_directions


ANALYSIS_VERSION = "crownwindow-v1"
MAX_BATCH_PIXELS = 19_000_000


class Capacity:
    """Bound expensive work for the local application."""

    def __init__(self, slots: int) -> None:
        self._available = BoundedSemaphore(slots)

    def __enter__(self):
        if not self._available.acquire(timeout=1.2):
            raise Problem("The local worker is busy with another observation. Try again shortly.", 429)
        return self

    def __exit__(self, kind, value, trace):
        self._available.release()
        return False


@dataclass(frozen=True)
class Order:
    photos: tuple[str, ...]
    lens_overrides: dict[str, str | None]
    footprint: Footprint
    settings: dict
    width: int
    height: int
    retain: bool
    name: str
    expected: str

    @classmethod
    def parse(cls, data: dict) -> Order:
        value = mapping(data, "Inspection request")
        fields(value, "photos", "lens_overrides", "footprint", "settings", "grid",
               "retain", "name", "expected")
        photos = tuple(identifier(raw, "Observation") for raw in items(value.get("photos"), "Observations", 1, 4))
        overrides = mapping(value.get("lens_overrides", {}), "Lens versions")
        versions = {identifier(photo): None if revision is None else identifier(revision, "Lens") for photo, revision in overrides.items()}
        width, height = items(value.get("grid", [160, 66]), "Grid shape", 2, 2)
        retain = boolean(value.get("retain", False), "Retain result")
        name = text(value.get("name"), "Result title", 85, default="")
        expected = text(value.get("expected"), "Preview fingerprint", 64, default="")
        if retain and (not name or len(expected) != 64 or any(x not in "0123456789abcdef" for x in expected)):
            raise Problem("Name this result and retain it from a current preview.")
        return cls(photos, versions, Footprint.parse(value.get("footprint")),
                   mapping(value.get("settings", {}), "Settings"),
                   whole(width, "Angular samples across", 36, 360),
                   whole(height, "Angular samples upward", 20, 150), retain, name, expected)

    def public(self) -> dict:
        return {"photos": list(self.photos), "lens_overrides": self.lens_overrides,
                "footprint": self.footprint.public(), "settings": self.settings,
                "grid": [self.width, self.height]}


@dataclass(frozen=True)
class Source:
    record: dict
    retained_lens: dict | None
    camera: Camera

    @classmethod
    def from_stored(cls, photo: dict, lens: dict | None, raw: bytes) -> Source:
        camera = Camera.read(raw, photo["image"]["file"])
        if camera.content_sha != photo["image"]["sha256"]:
            raise Problem("This camera observation no longer agrees with its recorded fingerprint.", 409)
        return cls(photo, lens, camera)

    @cached_property
    def lens(self) -> Lens:
        return Lens.parse(self.retained_lens["lens"]) if self.retained_lens else Lens.default()

    @cached_property
    def controls(self) -> Controls:
        return Controls.parse(self.record["classification"])

    @cached_property
    def classifier(self) -> Classifier:
        return Classifier.train(self.camera, self.lens, self.controls)

    @property
    def title(self) -> str:
        return self.record["capture"]["title"]

    @property
    def id(self) -> str:
        return self.record["id"]

    def sample(self, directions: Grid | np.ndarray) -> DirectionSamples:
        values = directions.directions if isinstance(directions, Grid) else np.asarray(directions)
        return sample_directions(self.camera, self.lens, self.classifier, values)

    def public(self) -> dict:
        return {
            "id": self.id,
            "capture": self.record["capture"],
            "image": self.record["image"],
            "lens_id": self.retained_lens["id"] if self.retained_lens else None,
            "lens": self.lens.public(),
            "lens_title": self.retained_lens["title"] if self.retained_lens else "Approximate full-circle lens",
            "classification": self.controls.public(),
            "source_revision": self.record["revision"],
            "original": self.record["original"],
            "preview": self.record["preview"],
        }

    def method_warnings(self) -> list[str]:
        facts = []
        if self.retained_lens is None:
            facts.append(self.title + " uses an unconfirmed full-circle lens placement.")
        if not any(region.kind == "sky" for region in self.controls.regions):
            facts.append(self.title + " uses built-in sky colors. Check the class view for cloud and glare errors.")
        if not any(region.kind == "obstruction" for region in self.controls.regions):
            facts.append(self.title + " uses built-in obstruction colors. Check the class view around pale leaves.")
        return facts


@dataclass(frozen=True)
class Bench:
    station: dict
    sources: tuple[Source, ...]
    order: Order

    @classmethod
    def load(cls, library: Library, station_id: str, order: Order) -> Bench:
        station, stored = library.acquire(station_id, list(order.photos), order.lens_overrides)
        size = sum(photo["image"]["width"] * photo["image"]["height"] for photo, _, _ in stored)
        if size > MAX_BATCH_PIXELS:
            raise Problem("The chosen observations exceed 19 million decoded pixels.", 413)
        return cls(station, tuple(Source.from_stored(*row) for row in stored), order)

    @cached_property
    def grid(self) -> Grid:
        return Grid.create(self.order.footprint, self.order.width, self.order.height)

    def one(self) -> Source:
        if len(self.sources) != 1:
            raise Problem("Select a single observation for this inspection.")
        return self.sources[0]

    def two(self) -> tuple[Source, Source]:
        if len(self.sources) != 2:
            raise Problem("Select two observations in comparison order.")
        return self.sources[0], self.sources[1]

    def make_source_identity(self) -> list[dict[str, Any]]:
        return [source.public() for source in self.sources]

    def fingerprint(self, tool: str) -> str:
        return key_of({"version": ANALYSIS_VERSION, "station": self.station["id"],
                       "tool": tool, "order": self.order.public(), "sources": self.make_source_identity()})

    def check_preview(self, tool: str) -> str:
        key = self.fingerprint(tool)
        if self.order.retain and self.order.expected != key:
            raise Problem("The lens, image settings, or inspection has changed. Run it again before retaining.", 409)
        return key
