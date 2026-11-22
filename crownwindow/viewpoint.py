"""A perspective view into a physically placed fisheye observation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .bench import Bench, Source
from .contracts import Problem, fields, number, real, whole
from .projection import directions_to_vectors, vectors_to_directions
from .result import Result
from .sky import MISSING, OPEN, state_counts
from .visual import GOLD, MINT, MUTED, PALE, _draw_font, comparison_panel, toned_raw


@dataclass(frozen=True)
class View:
    bearing: float
    elevation: float
    horizontal: float
    vertical: float
    width: int
    height: int

    @classmethod
    def parse(cls, raw: dict) -> View:
        fields(raw, "bearing", "elevation", "horizontal", "vertical", "width", "height")
        return cls(
            real(raw.get("bearing", 120), "View heading", -360, 720) % 360,
            real(raw.get("elevation", 29), "View elevation", -30, 90),
            real(raw.get("horizontal", 85), "Horizontal angle", 16, 135),
            real(raw.get("vertical", 54), "Vertical angle", 12, 122),
            whole(raw.get("width", 700), "Perspective width", 200, 1000),
            whole(raw.get("height", 445), "Perspective height", 140, 720),
        )

    @property
    def direction(self) -> np.ndarray:
        return np.array([self.bearing, self.elevation])

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = directions_to_vectors(self.direction)
        right = directions_to_vectors(np.array([self.bearing + 90., 0]))
        up = np.cross(right, center)
        return right, up, center

    def directions(self) -> np.ndarray:
        x = np.linspace(-1, 1, self.width) * np.tan(np.radians(self.horizontal / 2))
        y = np.linspace(1, -1, self.height) * np.tan(np.radians(self.vertical / 2))
        plane_x, plane_y = np.meshgrid(x, y)
        right, up, center = self.basis()
        field = center + plane_x[..., None] * right + plane_y[..., None] * up
        return vectors_to_directions(field)

    def ray(self, image: tuple[float, float]) -> np.ndarray:
        u, v = image
        x = (2 * u - 1) * np.tan(np.radians(self.horizontal / 2))
        y = (1 - 2 * v) * np.tan(np.radians(self.vertical / 2))
        right, up, center = self.basis()
        return vectors_to_directions(center + x * right + y * up)

    def to_view(self, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        right, up, center = self.basis()
        vectors = directions_to_vectors(directions)
        visible = vectors @ center
        with np.errstate(divide="ignore", invalid="ignore"):
            x = (vectors @ right) / visible / np.tan(np.radians(self.horizontal / 2))
            y = (vectors @ up) / visible / np.tan(np.radians(self.vertical / 2))
        united = np.stack((.5 + .5 * x, .5 - .5 * y), axis=-1)
        inside = (visible > 1e-6) & ((united >= 0) & (united <= 1)).all(axis=-1)
        return united, inside

    def public(self) -> dict:
        return self.__dict__


def _view_coordinates(view: View, source: Source):
    directions = view.directions()
    result = source.sample(directions)
    return result


def _points_report(view: View, observations) -> list[dict]:
    h, w = view.height, view.width
    points = {
        "top_left": (0, 0), "top_center": (0, w // 2), "top_right": (0, w - 1),
        "middle_left": (h // 2, 0), "center": (h // 2, w // 2), "middle_right": (h // 2, w - 1),
        "bottom_left": (h - 1, 0), "bottom_center": (h - 1, w // 2), "bottom_right": (h - 1, w - 1),
    }
    return [{"screen_mark": label, **observations.readout(index)} for label, index in points.items()]


def _plot_guides(view: View, image: np.ndarray) -> Image.Image:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    all_bear = np.arange(361)
    for elevation in (0, 15, 30, 45, 60, 75):
        world = np.stack((all_bear, np.full(361, elevation)), axis=-1)
        pixels, valid = view.to_view(world)
        _plot_curve(draw, pixels, valid, canvas.size, color=GOLD if elevation == 0 else MUTED)
    for heading, label in ((0, "N"), (45, "NE"), (90, "E"), (135, "SE"),
                           (180, "S"), (225, "SW"), (270, "W"), (315, "NW")):
        levels = np.arange(0, 90, 1)
        world = np.stack((np.full(len(levels), heading), levels), axis=-1)
        positions, valid = view.to_view(world)
        _plot_curve(draw, positions, valid, canvas.size, color=MINT)
        display = np.flatnonzero(valid)
        if not len(display):
            continue
        p = positions[display[min(3, len(display) - 1)]]
        draw.text((p[0] * (canvas.width - 1) + 4, p[1] * (canvas.height - 1) + 4), label,
                  fill=PALE, font=_draw_font(17), stroke_width=2, stroke_fill=(19, 31, 39))
    return canvas


def _plot_curve(draw, coordinates, inside, shape, color):
    width, height = shape
    for a, b, one, two in zip(coordinates[:-1], coordinates[1:], inside[:-1], inside[1:], strict=True):
        if one and two:
            draw.line(((a[0] * (width - 1), a[1] * (height - 1)),
                       (b[0] * (width - 1), b[1] * (height - 1))), fill=color, width=1)


def virtual_view(bench: Bench) -> Result:
    source = bench.one()
    view = View.parse(bench.order.settings)
    directions = _view_coordinates(view, source)
    pixels = toned_raw(directions.rgb, directions.measured)
    raw = Image.fromarray(pixels)
    with_guide = _plot_guides(view, pixels)
    map_report = _points_report(view, directions)
    result = Result("perspective", "Look toward a field direction",
                    "The perspective shows the selected bearing without rotating or modifying the camera original.")
    result.metrics = {
        "view": view.public(), "pixels_with_source_data": int(np.count_nonzero(directions.measured)),
        "pixels_looking_below_horizon": int(np.count_nonzero(directions.directions[..., 1] < 0)),
        "screen_pixel_classes": state_counts(directions.codes),
        "central_direction": [number(v, 4) for v in view.ray((.5, .5))],
    }
    result.readings = map_report
    result.png("direction-view", "Perspective camera evidence", raw)
    result.png("compass-view", "Bearing and elevation guide", with_guide)
    result.png("two-perspectives", "Camera rays and orientation", comparison_panel(
        [("As observed", raw), ("Compass and elevations", with_guide)], "View evidence"))
    result.csv("direction-inset-corners", "Screen direction map", map_report)
    result.warnings = [*source.method_warnings(),
                       "The viewpoint resamples camera pixels. A dark blue cell means the projection has no measured image support."]
    result.probe_kind = "perspective"
    return result


def perspective_probe(source: Source, view: View, positions: list[tuple[float, float]]) -> list[dict]:
    readings = []
    for index, point in enumerate(positions):
        ray = view.ray(point)
        item = {"screen_position": list(point), "direction": [number(v, 6) for v in ray]}
        if ray[1] < 0:
            item.update({"class": "below_horizon", "image_unit": None, "rgb": None, "sky_margin": None})
        else:
            sample = source.sample(ray[None, :])
            item.update(sample.readout(0))
        item["mark"] = index + 1
        readings.append(item)
    return readings
