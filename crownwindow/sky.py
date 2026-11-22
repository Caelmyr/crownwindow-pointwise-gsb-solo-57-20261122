"""Transparent sky/obstruction classification, native masks and spherical samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .camera import Camera, rgb_luminance, srgb_to_linear
from .contracts import Palette, Problem, enum, fields, items, mapping, number, point, real, text
from .projection import Grid, Lens


MISSING = 0
OCCLUDED = 1
OPEN = 2
UNCERTAIN = 3
LABELS = ("outside_measured_support", "obstructed", "open_sky", "uncertain")
SKY_COLORS = np.array([Palette().missing, Palette().blocked, Palette().sky, Palette().unknown], np.uint8)
DEFAULT_SKY = [(0.71, .82, .93), (.65, .75, .79), (.85, .87, .90), (.55, .69, .86)]
DEFAULT_OBSTACLES = [(.17, .29, .19), (.28, .35, .27), (.30, .28, .21), (.52, .6, .36)]


def sky_features(rgb: np.ndarray) -> np.ndarray:
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    value = rgb_luminance(srgb_to_linear(rgb))
    blue_minus_green = rgb[..., 2] - rgb[..., 1]
    neutral = (maximum - minimum) / np.maximum(maximum, .06)
    warm_axis = (rgb[..., 0] - rgb[..., 2]) * .85
    return np.stack((value, blue_minus_green, neutral * .45, warm_axis), axis=-1)


@dataclass(frozen=True)
class Region:
    label: str
    vertices: tuple[tuple[float, float], ...]
    kind: str

    @classmethod
    def parse(cls, raw: dict, index: int) -> Region:
        doc = mapping(raw, "Color region")
        fields(doc, "label", "vertices", "kind")
        vertices = tuple(point(p) for p in items(doc.get("vertices"), "Region vertices", 3, 50))
        if _area(vertices) < .000035:
            raise Problem(f"Color region {index + 1} is too small to sample.")
        return cls(text(doc.get("label") or f"Patch {index + 1}", "Region label", 70), vertices,
                   enum(doc.get("kind"), "Region type", ("sky", "obstruction", "exclude")))

    def public(self) -> dict:
        return {"label": self.label, "vertices": self.vertices, "kind": self.kind}

    def mask(self, camera: Camera) -> np.ndarray:
        canvas = Image.new("1", (camera.width, camera.height))
        points = [tuple(p) for p in camera.unit_to_pixel(self.vertices)]
        ImageDraw.Draw(canvas).polygon(points, fill=1)
        return np.asarray(canvas, dtype=bool)

    def contains(self, uv: np.ndarray) -> np.ndarray:
        x, y = uv[..., 0], uv[..., 1]
        result = np.zeros(x.shape, dtype=bool)
        points = self.vertices
        for (a, b), (c, d) in zip(points, points[1:] + points[:1], strict=True):
            if abs(d - b) < 1e-9:
                continue
            changes_side = (b > y) != (d > y)
            crossing = a + (c - a) * (y - b) / (d - b)
            result ^= changes_side & (x < crossing)
        return result


def _area(vertices: tuple[tuple[float, float], ...]) -> float:
    return abs(sum(a * d - c * b for (a, b), (c, d) in zip(vertices, vertices[1:] + vertices[:1], strict=True))) / 2


@dataclass(frozen=True)
class Controls:
    margin: float
    distance_ceiling: float
    clipping_guard: float
    regions: tuple[Region, ...]

    @classmethod
    def parse(cls, raw: dict | None) -> Controls:
        doc = mapping(raw or {}, "Sky classifier")
        fields(doc, "margin", "distance_ceiling", "clipping_guard", "regions")
        return cls(
            real(doc.get("margin", .16), "Ambiguity margin", .01, .6),
            real(doc.get("distance_ceiling", .48), "Color confidence limit", .10, 1.2),
            real(doc.get("clipping_guard", .997), "Clipping guard", .85, 1),
            tuple(Region.parse(raw, i) for i, raw in enumerate(items(doc.get("regions", []), "Color samples", 0, 24))),
        )

    def public(self) -> dict:
        return {"margin": self.margin, "distance_ceiling": self.distance_ceiling,
                "clipping_guard": self.clipping_guard, "regions": [r.public() for r in self.regions]}


@dataclass(frozen=True)
class Classifier:
    sky: np.ndarray
    obstructed: np.ndarray
    options: Controls
    proof: tuple[dict, ...]

    @classmethod
    def train(cls, camera: Camera, lens: Lens, options: Controls) -> Classifier:
        sky = []
        obstruction = []
        reports = []
        exclusions = [r for r in options.regions if r.kind == "exclude"]
        for region in options.regions:
            if region.kind == "exclude":
                reports.append({"label": region.label, "role": region.kind, "kept": None})
                continue
            counts = _training_pixels(region, camera, lens, exclusions)
            if len(counts) < 12:
                raise Problem(f"{region.label} has too few supported pixels in the lens.")
            features = sky_features(counts)
            prototypes = np.quantile(features, [.28, .52, .74], axis=0)
            destination = sky if region.kind == "sky" else obstruction
            destination.extend(prototypes)
            reports.append({
                "label": region.label,
                "role": region.kind,
                "kept": len(counts),
                "median_rgb": [number(v, 5) for v in np.median(counts, axis=0)],
                "median_features": [number(v, 5) for v in prototypes[1]],
            })
        if not sky:
            sky = sky_features(np.array(DEFAULT_SKY, np.float32))
        if not obstruction:
            obstruction = sky_features(np.array(DEFAULT_OBSTACLES, np.float32))
        return cls(np.asarray(sky), np.asarray(obstruction), options, tuple(reports))

    def _confidence(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = features.shape[:-1]
        distances_to_sky = np.full(shape, np.inf)
        distances_to_obstruction = np.full(shape, np.inf)
        weights = np.array([1.1, 1.3, 1, .9])
        for reference in self.sky:
            d = np.sum(((features - reference) * weights) ** 2, axis=-1)
            distances_to_sky = np.minimum(d, distances_to_sky)
        for reference in self.obstructed:
            d = np.sum(((features - reference) * weights) ** 2, axis=-1)
            distances_to_obstruction = np.minimum(d, distances_to_obstruction)
        sky = np.sqrt(distances_to_sky)
        obstruction = np.sqrt(distances_to_obstruction)
        contrast = (obstruction - sky) / np.maximum(obstruction + sky, 1e-7)
        return sky, obstruction, contrast

    def classify(self, colors: np.ndarray, measured: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Classify measured evidence. Exclusions remain non-measured."""
        features = sky_features(colors)
        sky, obstruction, contrast = self._confidence(features)
        support = measured.copy()
        for region in self.options.regions:
            if region.kind == "exclude":
                support &= ~region.contains(uv)
        closest = np.minimum(sky, obstruction)
        upper_bound = np.max(colors, axis=-1) >= self.options.clipping_guard
        little_signal = np.max(colors, axis=-1) < .018
        weak = upper_bound | little_signal | (closest > self.options.distance_ceiling)
        tight = np.abs(contrast) < self.options.margin
        sky_guess = np.where(contrast >= 0, OPEN, OCCLUDED)
        codes = np.where(support, np.where(weak | tight, UNCERTAIN, sky_guess), MISSING)
        score = np.where(support, contrast, np.nan)
        return codes.astype(np.uint8), score.astype(np.float32)

    def describe(self) -> dict:
        has_sky = any(r.kind == "sky" for r in self.options.regions)
        has_obstructed = any(r.kind == "obstruction" for r in self.options.regions)
        return {
            "settings": self.options.public(),
            "samples": list(self.proof),
            "built_in_sky_colors": not has_sky,
            "built_in_obstruction_colors": not has_obstructed,
            "classes": list(LABELS),
        }


def _training_pixels(region: Region, camera: Camera, lens: Lens, exclusions: list[Region]) -> np.ndarray:
    mask = region.mask(camera)
    mask = mask & camera.valid
    y, x = np.nonzero(mask)
    stride = max(1, len(y) // 16500)
    y, x = y[::stride], x[::stride]
    points = camera.pixel_to_unit(np.stack((x, y), axis=-1))
    _, inside = lens.image_to_world(points, camera)
    for exclusion in exclusions:
        inside &= ~exclusion.contains(points)
    colors = camera.color[y, x]
    inside &= (colors.max(axis=-1) < .998)
    inside &= (colors.max(axis=-1) > .02)
    return colors[inside]


@dataclass(frozen=True)
class DirectionSamples:
    directions: np.ndarray
    native: np.ndarray
    rgb: np.ndarray
    codes: np.ndarray
    margin: np.ndarray

    @property
    def measured(self) -> np.ndarray:
        return self.codes > MISSING

    @property
    def known(self) -> np.ndarray:
        return (self.codes == OPEN) | (self.codes == OCCLUDED)

    def readout(self, index: AnyIndex) -> dict:
        point = self.native[index]
        direction = self.directions[index]
        code = int(self.codes[index])
        return {
            "direction": [number(q, 5) for q in direction],
            "image_unit": [number(q, 6) for q in point] if code else None,
            "class": LABELS[code],
            "rgb": [number(q, 5) for q in self.rgb[index]] if code else None,
            "sky_margin": number(self.margin[index], 6),
        }


AnyIndex = int | tuple[int, ...]


def sample_directions(camera: Camera, lens: Lens, classifier: Classifier,
                      directions: np.ndarray) -> DirectionSamples:
    points, geometry = lens.world_to_image(directions, camera)
    value, native_support = camera.bilinear(points)
    codes, score = classifier.classify(value, geometry & native_support, points)
    withheld = lens.withheld(directions)
    if np.any(withheld):
        codes = codes.copy()
        codes[withheld] = MISSING
        score = score.copy()
        score[withheld] = np.nan
    return DirectionSamples(directions, points, value, codes, score)


def geographical_classes(camera: Camera, lens: Lens, classifier: Classifier, grid: Grid) -> DirectionSamples:
    return sample_directions(camera, lens, classifier, grid.directions)


def native_classes(camera: Camera, lens: Lens, classifier: Classifier, edge: int = 720) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratio = min(1., edge / max(camera.width, camera.height))
    nx, ny = int(round(camera.width * ratio)), int(round(camera.height * ratio))
    x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    points = np.stack((x, y), axis=-1)
    rgb, observed = camera.bilinear(points)
    _, in_lens = lens.image_to_world(points, camera)
    codes, contrast = classifier.classify(rgb, observed & in_lens, points)
    return rgb, codes, contrast


def state_counts(codes: np.ndarray, weights: np.ndarray | float = 1.) -> dict:
    values = np.bincount(np.asarray(codes).reshape(-1), minlength=len(LABELS))
    if isinstance(weights, np.ndarray):
        weight = np.broadcast_to(weights, codes.shape)
        area = np.bincount(np.asarray(codes).reshape(-1), weight.reshape(-1), minlength=len(LABELS))
    else:
        area = values * weights
    return {label: {"count": int(count), "weight": number(a, 8)}
            for label, count, a in zip(LABELS, values, area, strict=True)}
