"""Fit and audit a level fisheye lens against marked field directions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .camera import Camera
from .contracts import Problem, angle, fields, items, mapping, number, point, real, text
from .projection import Lens, angular_distance, bearing_delta


@dataclass(frozen=True)
class Landmark:
    title: str
    image: tuple[float, float]
    direction: tuple[float, float]
    confidence: float

    @classmethod
    def parse(cls, raw: dict, index: int) -> Landmark:
        value = mapping(raw, "Landmark")
        fields(value, "title", "image", "direction", "confidence")
        return cls(
            text(value.get("title") or f"Field mark {index + 1}", "Landmark title", 65),
            point(value.get("image"), "Image position"),
            angle(value.get("direction"), "Survey direction"),
            real(value.get("confidence", 1), "Landmark confidence", .05, 1),
        )

    def public(self) -> dict:
        return {"title": self.title, "image": list(self.image),
                "direction": list(self.direction), "confidence": self.confidence}


@dataclass(frozen=True)
class Calibration:
    lens: Lens
    landmarks: tuple[Landmark, ...]
    rows: tuple[dict, ...]
    initial: Lens
    metrics: dict
    mode: str

    def public(self) -> dict:
        return {
            "lens": self.lens.public(),
            "landmarks": [mark.public() for mark in self.landmarks],
            "residuals": list(self.rows),
            "initial": self.initial.public(),
            "metrics": self.metrics,
            "mode": self.mode,
        }


def _from_parameters(vector: np.ndarray, template: Lens) -> Lens:
    cx, cy, radius, north = vector
    return Lens((float(cx), float(cy)), float(radius), template.maximum_zenith,
                float(north) % 360, template.model)


def _landmark_arrays(landmarks: tuple[Landmark, ...], camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = camera.unit_to_pixel([mark.image for mark in landmarks])
    world = np.array([mark.direction for mark in landmarks], dtype=float)
    weights = np.sqrt(np.array([mark.confidence for mark in landmarks]))
    return points, world, weights


def _general_position(pixels: np.ndarray, directions: np.ndarray, camera: Camera) -> None:
    shifted = pixels - pixels.mean(axis=0)
    spans = np.linalg.svd(shifted, compute_uv=False)
    if spans[1] < max(camera.width, camera.height) * .018 or spans[1] * 4 < spans[0]:
        raise Problem("Spread the marks in two image directions before fitting the lens.")
    elevations = directions[:, 1]
    if np.any(elevations >= 89.99):
        raise Problem("Replace the zenith mark with a nearby landmark with a definite bearing.")
    for i, direction in enumerate(directions[:-1]):
        other = directions[i + 1:]
        if np.any(angular_distance(other, direction) < .14):
            raise Problem("Two landmarks give effectively the same field direction.")


def _error_function(vector, template, camera, pixels, world, weights) -> np.ndarray:
    lens = _from_parameters(vector, template)
    trial, _ = lens.project_pixels(world, camera)
    return ((trial - pixels) * weights[:, None]).reshape(-1)


def _marks_from(raw: list) -> tuple[Landmark, ...]:
    return tuple(Landmark.parse(entry, index) for index, entry in enumerate(items(raw, "Landmarks", 4, 30)))


def fit_lens(camera: Camera, initial: Lens, raw_landmarks: list) -> Calibration:
    landmarks = _marks_from(raw_landmarks)
    pixels, world, weights = _landmark_arrays(landmarks, camera)
    _general_position(pixels, world, camera)
    if np.any((90 - world[:, 1]) > initial.maximum_zenith):
        raise Problem("Some surveyed marks fall beyond the indicated lens radius.")
    dimensions = np.array([*initial.center_uv, initial.radius_over_short_side])
    north_residual = _initial_bearing(camera, pixels, world, initial)
    x0 = np.array([*dimensions, initial.north_clockwise + north_residual])
    lower = [0, 0, .065, x0[3] - 180]
    upper = [1, 1, 2.5, x0[3] + 180]
    result = least_squares(_error_function, np.clip(x0, lower, upper),
                           bounds=(lower, upper),
                           args=(initial, camera, pixels, world, weights),
                           x_scale=[.3, .3, .35, 110],
                           f_scale=max(2, min(camera.width, camera.height) / 215),
                           loss="soft_l1", max_nfev=160)
    if not result.success:
        raise Problem("Could not settle on a stable lens fit. Check the markings.")
    if np.linalg.matrix_rank(result.jac, tol=.05) < 4:
        raise Problem("The surveyed landmarks do not constrain all lens settings.")
    lens = _from_parameters(result.x, initial)
    return examine_fit(camera, initial, lens, landmarks)


def _initial_bearing(camera: Camera, pixels: np.ndarray, world: np.ndarray, initial: Lens) -> float:
    offset = pixels - camera.unit_to_pixel(initial.center_uv)
    lengths = np.linalg.norm(offset, axis=1)
    apparent = np.degrees(np.arctan2(offset[:, 0], -offset[:, 1]))
    offsets = bearing_delta(apparent, world[:, 0] + initial.north_clockwise)
    radians = np.radians(offsets)
    sin_sum = np.sum(lengths * np.sin(radians))
    cos_sum = np.sum(lengths * np.cos(radians))
    return float(np.degrees(np.arctan2(sin_sum, cos_sum)))


def examine_fit(camera: Camera, initial: Lens, lens: Lens, landmarks: tuple[Landmark, ...]) -> Calibration:
    pixels, world, _ = _landmark_arrays(landmarks, camera)
    predicted, _ = lens.project_pixels(world, camera)
    initial_pixels, _ = initial.project_pixels(world, camera)
    world_at_points, in_view = lens.image_to_world(camera.pixel_to_unit(pixels), camera)
    pixel_error = np.linalg.norm(predicted - pixels, axis=1)
    initial_error = np.linalg.norm(initial_pixels - pixels, axis=1)
    angle_error = angular_distance(world_at_points, world)
    median = np.median(pixel_error)
    spread = np.median(np.abs(pixel_error - median)) * 1.4826
    limit = max(2.5, median + max(3 * spread, 2))
    rows = []
    for index, mark in enumerate(landmarks):
        rows.append({
            "mark": mark.title,
            "known_direction": list(mark.direction),
            "entered_image": list(mark.image),
            "predicted_image": [number(v, 6) for v in camera.pixel_to_unit(predicted[index])],
            "residual_px": number(pixel_error[index], 3),
            "initial_residual_px": number(initial_error[index], 3),
            "residual_degrees": number(angle_error[index], 4) if in_view[index] else None,
            "review_needed": bool(pixel_error[index] > limit or not in_view[index]),
        })
    return Calibration(lens, landmarks, tuple(rows), initial, {
        "mark_count": len(landmarks),
        "rms_px": number(np.sqrt(np.mean(pixel_error ** 2)), 3),
        "initial_rms_px": number(np.sqrt(np.mean(initial_error ** 2)), 3),
        "median_px": number(median, 3),
        "largest_px": number(pixel_error.max(), 3),
        "review_count": sum(row["review_needed"] for row in rows),
    }, "field_landmarks")


def lens_limits(lens: Lens, camera: Camera) -> dict:
    bearings = np.arange(0, 360, 1.)
    elevations = np.arange(0, 91, 1.)
    b, e = np.meshgrid(bearings, elevations)
    projected, inside = lens.world_to_image(np.stack((b, e), axis=-1), camera)
    _, observed = camera.bilinear(np.clip(projected, 0, 1))
    actual = inside & observed
    geom_direction = inside.any(axis=0)
    report = []
    for azimuth in range(0, 360, 30):
        present = np.flatnonzero(actual[:, azimuth])
        geometric = np.flatnonzero(inside[:, azimuth])
        report.append({
            "bearing": azimuth,
            "lowest_geometry": int(geometric[0]) if len(geometric) else None,
            "lowest_observed": int(present[0]) if len(present) else None,
            "highest_observed": int(present[-1]) if len(present) else None,
            "distinct_observed_bands": int(np.sum(np.diff(np.r_[False, actual[:, azimuth], False].astype(int)) > 0)),
        })
    area_by_elevation = np.cos(np.radians(elevations))
    shape = (len(elevations), len(bearings))
    expected = np.broadcast_to(area_by_elevation[:, None], shape).sum()
    return {"lens": lens.public(), "rows": report,
            "zenith_in_image": bool(geom_direction.any() and inside[-1, 0]),
            "approximate_spherical_geometry_share": number((inside * area_by_elevation[:, None]).sum() / expected),
            "approximate_spherical_pixel_support_share": number((actual * area_by_elevation[:, None]).sum() / expected)}
