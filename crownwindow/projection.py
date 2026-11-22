"""Fisheye placement and unit-sphere direction calculations."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import numpy as np

from .camera import Camera
from .contracts import Footprint, Problem, angle, enum, fields, items, mapping, number, point, real


MODELS = ("equidistant", "equisolid", "orthographic", "stereographic")
LEAN_LIMIT = 12.0


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


def _rodrigues(alpha: float, n: np.ndarray) -> np.ndarray:
    x, y, z = n
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(alpha) * skew + (1 - np.cos(alpha)) * skew @ skew


def _tilt_axis(tilt_bearing: float) -> np.ndarray:
    return np.array([np.cos(np.radians(tilt_bearing)), -np.sin(np.radians(tilt_bearing)), 0.0])


def world_to_camera(vectors: np.ndarray, lean: tuple[float, float]) -> np.ndarray:
    """Rotate local ENU unit vectors into the tilted camera frame."""
    tilt, tilt_bearing = lean
    if abs(tilt) < 1e-10:
        return np.asarray(vectors, dtype=float)
    return np.asarray(vectors, dtype=float) @ _rodrigues(np.radians(tilt), _tilt_axis(tilt_bearing)).T


def camera_to_world(vectors: np.ndarray, lean: tuple[float, float]) -> np.ndarray:
    tilt, tilt_bearing = lean
    if abs(tilt) < 1e-10:
        return np.asarray(vectors, dtype=float)
    return np.asarray(vectors, dtype=float) @ _rodrigues(-np.radians(tilt), _tilt_axis(tilt_bearing)).T


def parse_lean(raw) -> tuple[float, float]:
    if raw is None:
        return 0.0, 0.0
    data = mapping(raw, "Optical axis tilt")
    fields(data, "tilt_degrees", "tilt_bearing")
    tilt = real(data.get("tilt_degrees", 0), "Optical axis tilt", 0, LEAN_LIMIT)
    bearing = real(data.get("tilt_bearing", 0), "Tilt bearing", -360, 720) % 360
    return tilt, bearing


@dataclass(frozen=True)
class RasterHoldout:
    """Directions withheld from measurement on a fixed equal-angle grid."""

    columns: int
    rows: int
    packed: bytes

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> "RasterHoldout":
        rows, columns = mask.shape
        flat = np.asarray(mask, dtype=bool).reshape(-1)
        padded = np.concatenate((flat, np.zeros((-len(flat)) % 8, dtype=bool)))
        bits = padded.reshape(-1, 8)
        values = np.sum(bits * (1 << np.arange(7, -1, -1)), axis=1).astype(np.uint8)
        return cls(columns, rows, values.tobytes())

    @classmethod
    def parse(cls, raw) -> "RasterHoldout | None":
        if raw is None:
            return None
        data = mapping(raw, "Angular holdout")
        fields(data, "columns", "rows", "mask")
        columns = int(real(data.get("columns"), "Holdout columns", 8, 720))
        rows = int(real(data.get("rows"), "Holdout rows", 4, 180))
        text_value = data.get("mask")
        if not isinstance(text_value, str):
            raise Problem("The holdout mask must be base64 packed bits.")
        try:
            packed = base64.b64decode(text_value, validate=True)
        except ValueError as error:
            raise Problem("The holdout mask is not valid base64.") from error
        return cls(columns, rows, packed)

    @property
    def mask(self) -> np.ndarray:
        bits = np.unpackbits(np.frombuffer(self.packed, dtype=np.uint8))
        return bits[:self.columns * self.rows].reshape(self.rows, self.columns)

    def public(self) -> dict:
        return {"columns": self.columns, "rows": self.rows,
                "mask": base64.b64encode(self.packed).decode("ascii")}

    def withheld(self, directions: np.ndarray) -> np.ndarray:
        bearing, elevation = np.moveaxis(np.asarray(directions, float), -1, 0)
        col = np.floor(bearing / 360 * self.columns).astype(int) % self.columns
        row = np.clip(np.floor(elevation / 90 * self.rows).astype(int), 0, self.rows - 1)
        flat = np.minimum(row * self.columns + col, self.columns * self.rows - 1)
        return np.unpackbits(np.frombuffer(self.packed, dtype=np.uint8))[flat].astype(bool)

    def cell_count(self) -> int:
        return int(np.count_nonzero(self.mask))


@dataclass(frozen=True)
class Lens:
    """A radial lens law, an image bearing and an optional optical-axis tilt."""

    center_uv: tuple[float, float]
    radius_over_short_side: float
    maximum_zenith: float
    north_clockwise: float
    model: str
    lean: tuple[float, float] = (0.0, 0.0)
    holdout: RasterHoldout | None = None

    @classmethod
    def parse(cls, value: dict) -> Lens:
        data = mapping(value, "Lens placement")
        fields(data, "center_uv", "radius_over_short_side", "maximum_zenith", "north_clockwise",
               "model", "lean", "holdout")
        return cls(
            point(data.get("center_uv", [.5, .5]), "Lens center"),
            real(data.get("radius_over_short_side", .49), "Lens radius", .065, 2.5),
            real(data.get("maximum_zenith", 90), "Maximum angle from zenith", 10, 90),
            real(data.get("north_clockwise", 0), "Camera north rotation", -720, 720) % 360,
            enum(data.get("model", "equidistant"), "Projection", MODELS),
            parse_lean(data.get("lean")),
            RasterHoldout.parse(data.get("holdout")),
        )

    @classmethod
    def default(cls) -> Lens:
        return cls((.5, .5), .49, 90, 0, "equidistant")

    @property
    def tilted(self) -> bool:
        return bool(self.lean[0] > 1e-9)

    def public(self) -> dict:
        return {
            "center_uv": list(self.center_uv),
            "radius_over_short_side": self.radius_over_short_side,
            "maximum_zenith": self.maximum_zenith,
            "north_clockwise": self.north_clockwise,
            "model": self.model,
            "lean": {"tilt_degrees": number(self.lean[0], 6), "tilt_bearing": number(self.lean[1], 6)},
            "holdout": self.holdout.public() if self.holdout else None,
        }

    def scale_px(self, camera: Camera) -> float:
        return min(camera.width - 1, camera.height - 1) * self.radius_over_short_side

    def radial_fraction(self, zenith: np.ndarray) -> np.ndarray:
        return _radial(np.radians(zenith), self.model) / _radial(np.radians(self.maximum_zenith), self.model)

    def project_pixels(self, direction: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
        vectors = directions_to_vectors(direction)
        camera_vectors = world_to_camera(vectors, self.lean)
        cx, cy, cz = np.moveaxis(camera_vectors, -1, 0)
        theta = np.arccos(np.clip(cz, -1, 1))
        azimuth_in_camera = np.arctan2(cx, cy) + np.radians(self.north_clockwise)
        radial = self.radial_fraction(np.degrees(theta)) * self.scale_px(camera)
        center = camera.unit_to_pixel(self.center_uv)
        projected = np.stack((center[0] + radial * np.sin(azimuth_in_camera),
                              center[1] - radial * np.cos(azimuth_in_camera)), axis=-1)
        geometrical = np.isfinite(radial) & (theta >= -1e-7) & (theta <= np.radians(self.maximum_zenith) + 1e-7)
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
        azimuth = np.arctan2(dx, -dy) - np.radians(self.north_clockwise)
        camera_vectors = np.stack((np.sin(theta) * np.sin(azimuth),
                                   np.sin(theta) * np.cos(azimuth),
                                   np.cos(theta)), axis=-1)
        x, y, z = np.moveaxis(camera_to_world(camera_vectors, self.lean), -1, 0)
        bearing = np.degrees(np.arctan2(x, y)) % 360
        elevation = np.degrees(np.arctan2(z, np.hypot(x, y)))
        inside = possible & (fraction <= 1 + 1e-7) & (elevation >= -1e-7)
        in_frame = ((np.asarray(point) >= -1e-7) & (np.asarray(point) <= 1 + 1e-7)).all(axis=-1)
        return np.stack((bearing, elevation), axis=-1), inside & in_frame

    def withheld(self, directions: np.ndarray) -> np.ndarray:
        if self.holdout is None:
            return np.zeros(np.asarray(directions).shape[:-1], dtype=bool)
        return self.holdout.withheld(directions)

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
