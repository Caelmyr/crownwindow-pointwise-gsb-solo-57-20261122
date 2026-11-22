"""Fisheye placement and unit-sphere direction calculations."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .camera import Camera
from .contracts import Footprint, Problem, angle, enum, fields, items, mapping, number, point, real


MODELS = ("equidistant", "equisolid", "orthographic", "stereographic")


def _radial(theta: np.ndarray, model: str) -> np.ndarray:
    if model == "equidistant":
        return theta
    if model == "equisolid":
        return 2 * np.sin(theta / 2)
    if model == "orthographic":
        return np.sin(theta)
    return 2 * np.tan(theta / 2)


def _theta(distance: np.ndarray, model: str) -> tuple[np.ndarray, np.ndarray]:
    distance = np.asarray(distance, dtype=float)
    if model == "equidistant":
        return distance, np.isfinite(distance)
    if model == "equisolid":
        return 2 * np.arcsin(np.clip(distance / 2, -1, 1)), np.abs(distance / 2) <= 1
    if model == "orthographic":
        return np.arcsin(np.clip(distance, -1, 1)), np.abs(distance) <= 1
    return 2 * np.arctan(distance / 2), np.isfinite(distance)


def angular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    va = directions_to_vectors(a)
    vb = directions_to_vectors(b)
    cross = np.linalg.norm(np.cross(va, vb), axis=-1)
    dot = (va * vb).sum(axis=-1)
    return np.degrees(np.arctan2(cross, dot))


def directions_to_vectors(direction: np.ndarray) -> np.ndarray:
    bearing, elevation = np.moveaxis(np.radians(direction), -1, 0)
    footprint = np.cos(elevation)
    return np.stack((footprint * np.sin(bearing), footprint * np.cos(bearing), np.sin(elevation)), axis=-1)


def vectors_to_directions(vectors: np.ndarray) -> np.ndarray:
    x, y, z = np.moveaxis(vectors, -1, 0)
    return np.stack((np.degrees(np.arctan2(x, y)) % 360, np.degrees(np.arctan2(z, np.hypot(x, y)))), axis=-1)


def interpolate_arc(origin: tuple[float, float], end: tuple[float, float], inside: np.ndarray) -> np.ndarray:
    if len(inside) < 1:
        return np.empty((0, 2))
    a, b = directions_to_vectors(np.array([origin, end]))
    omega = np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))
    if abs(np.sin(omega)) < 1e-7:
        if omega > np.pi / 2:
            raise Problem("An arc through opposite directions needs a midpoint.")
        vector = (1 - inside[:, None]) * a + inside[:, None] * b
    else:
        vector = (np.sin((1 - inside) * omega)[:, None] * a + np.sin(inside * omega)[:, None] * b)
    return vectors_to_directions(vector)


def bearing_delta(a: float | np.ndarray, b: float | np.ndarray) -> np.ndarray:
    return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180


@dataclass(frozen=True)
class Lens:
    """A level optical axis, a radial lens law, and an image bearing."""

    center_uv: tuple[float, float]
    radius_over_short_side: float
    maximum_zenith: float
    north_clockwise: float
    model: str

    @classmethod
    def parse(cls, value: dict) -> Lens:
        data = mapping(value, "Lens placement")
        fields(data, "center_uv", "radius_over_short_side", "maximum_zenith", "north_clockwise", "model")
        return cls(
            point(data.get("center_uv", [.5, .5]), "Lens center"),
            real(data.get("radius_over_short_side", .49), "Lens radius", .065, 2.5),
            real(data.get("maximum_zenith", 90), "Maximum angle from zenith", 10, 90),
            real(data.get("north_clockwise", 0), "Camera north rotation", -720, 720) % 360,
            enum(data.get("model", "equidistant"), "Projection", MODELS),
        )

    @classmethod
    def default(cls) -> Lens:
        return cls((.5, .5), .49, 90, 0, "equidistant")

    def public(self) -> dict:
        return {
            "center_uv": list(self.center_uv),
            "radius_over_short_side": self.radius_over_short_side,
            "maximum_zenith": self.maximum_zenith,
            "north_clockwise": self.north_clockwise,
            "model": self.model,
        }

    def scale_px(self, camera: Camera) -> float:
        return min(camera.width - 1, camera.height - 1) * self.radius_over_short_side

    def radial_fraction(self, zenith: np.ndarray) -> np.ndarray:
        return _radial(np.radians(zenith), self.model) / _radial(np.radians(self.maximum_zenith), self.model)

    def project_pixels(self, direction: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
        bearing, elevation = np.moveaxis(np.asarray(direction, dtype=float), -1, 0)
        theta = 90 - elevation
        azimuth_in_camera = np.radians(bearing + self.north_clockwise)
        radial = self.radial_fraction(theta) * self.scale_px(camera)
        center = camera.unit_to_pixel(self.center_uv)
        projected = np.stack((center[0] + radial * np.sin(azimuth_in_camera),
                              center[1] - radial * np.cos(azimuth_in_camera)), axis=-1)
        geometrical = np.isfinite(radial) & (theta >= -1e-7) & (theta <= self.maximum_zenith + 1e-7)
        return projected, geometrical

    def world_to_image(self, direction: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
        pixels, inside = self.project_pixels(direction, camera)
        position = camera.pixel_to_unit(pixels)
        return position, inside & ((position >= -1e-7) & (position <= 1 + 1e-7)).all(axis=-1)

    def image_to_world(self, point: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
        pixel = camera.unit_to_pixel(point)
        center = camera.unit_to_pixel(self.center_uv)
        offsets = pixel - center
        dx, dy = offsets[..., 0], offsets[..., 1]
        fraction = np.hypot(dx, dy) / self.scale_px(camera)
        d = fraction * _radial(np.radians(self.maximum_zenith), self.model)
        theta, possible = _theta(d, self.model)
        bearing = (np.degrees(np.arctan2(dx, -dy)) - self.north_clockwise) % 360
        elevation = 90 - np.degrees(theta)
        inside = possible & (fraction <= 1 + 1e-7) & (elevation >= -1e-7)
        in_frame = ((np.asarray(point) >= -1e-7) & (np.asarray(point) <= 1 + 1e-7)).all(axis=-1)
        return np.stack((bearing, elevation), axis=-1), inside & in_frame

    def readout(self, camera: Camera, uv: tuple[float, float]) -> dict:
        direction, inside = self.image_to_world(np.array(uv), camera)
        point_readout = camera.read_nearest(uv)
        point_readout.update({
            "direction": [number(direction[0], 4), number(direction[1], 4)] if inside else None,
            "in_lens": bool(inside),
            "pixel_spanning_zenith": bool(np.linalg.norm(camera.unit_to_pixel(uv) -
                                                       camera.unit_to_pixel(self.center_uv)) < .5),
        })
        return point_readout


@dataclass(frozen=True)
class Grid:
    """Angular cells with equal solid-angle weights inside a footprint."""

    bearing_edges: np.ndarray
    sine_edges: np.ndarray
    directions: np.ndarray
    footprint: Footprint
    solid_angle_per_cell: float

    @classmethod
    def create(cls, footprint: Footprint, columns: int, rows: int) -> Grid:
        lower = np.sin(np.radians(footprint.bottom))
        upper = np.sin(np.radians(footprint.top))
        sine_edges = np.linspace(lower, upper, rows + 1)
        azimuth_edges = np.linspace(footprint.bearing, footprint.bearing + footprint.sweep, columns + 1)
        s = (sine_edges[:-1] + sine_edges[1:]) / 2
        b = (azimuth_edges[:-1] + azimuth_edges[1:]) / 2
        bearings, elevations = np.meshgrid(b % 360, np.degrees(np.arcsin(s)))
        return cls(azimuth_edges, sine_edges, np.stack((bearings, elevations), axis=-1),
                   footprint, footprint.steradians / (columns * rows))

    @property
    def shape(self) -> tuple[int, int]:
        return self.directions.shape[:2]

    @property
    def periodic(self) -> bool:
        return abs(self.footprint.sweep - 360) < 1e-7

    @property
    def cell_elevations(self) -> np.ndarray:
        return np.degrees(np.arcsin(self.sine_edges))

    def index(self, direction: tuple[float, float]) -> tuple[int, int] | None:
        bearing, elevation = direction
        offset = (bearing - self.footprint.bearing) % 360
        sine = np.sin(np.radians(elevation))
        if offset > self.footprint.sweep or sine < self.sine_edges[0] or sine > self.sine_edges[-1]:
            return None
        row = int(np.searchsorted(self.sine_edges, sine, side="right") - 1)
        col = int(np.searchsorted(self.bearing_edges, self.footprint.bearing + offset, side="right") - 1)
        return min(row, self.shape[0] - 1), min(col, self.shape[1] - 1)

    def member_intervals(self, row: int, col: int) -> dict:
        elevations = self.cell_elevations
        return {"elevation": [number(elevations[row]), number(elevations[row + 1])],
                "bearing": [number(self.bearing_edges[col] % 360), number(self.bearing_edges[col + 1] % 360)],
                "crosses_north": bool(self.bearing_edges[col] // 360 != self.bearing_edges[col + 1] // 360)}

    def sample_offsets(self, factor: int) -> np.ndarray:
        rows, cols = self.shape
        offsets = (np.arange(factor) + .5) / factor
        delta_s = (self.sine_edges[1] - self.sine_edges[0]) * offsets
        delta_b = (self.bearing_edges[1] - self.bearing_edges[0]) * offsets
        s = self.sine_edges[:-1, None] + delta_s
        b = self.bearing_edges[:-1, None] + delta_b
        az, el = np.meshgrid(b.reshape(cols * factor) % 360, np.degrees(np.arcsin(s.reshape(rows * factor))))
        return np.stack((az, el), axis=-1)

    def describe(self) -> dict:
        return {"footprint": self.footprint.public(), "columns": self.shape[1], "rows": self.shape[0],
                "cell_solid_angle_sr": number(self.solid_angle_per_cell, 8)}


def world_points(value: list, minimum: int = 1, maximum: int = 80) -> list[tuple[float, float]]:
    return [angle(x) for x in items(value, "Directions", minimum, maximum)]
