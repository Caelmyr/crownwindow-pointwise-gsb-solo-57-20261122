"""Tilted optical axis placement from surveyed corners and two-fit agreement.

A level :class:`~crownwindow.projection.Lens` assumes its optical axis points at
the zenith.  When the camera was not levelled the same radial law still applies
inside the camera frame, but that frame is rotated relative to the local
vertical.  The lens ``lean`` describes that rotation as a tilt magnitude and the
world bearing toward which the optical axis leans.

Two placements are always offered together:

* ``fixed_circle`` keeps the previously measured circle (center, radius, radial
  law) and solves only the tilt and image bearing;
* ``adjusted_circle`` lets the center and radius move slightly as well.

Where the two recovered placements disagree in direction, the belt is not
trustworthy enough for a sky share and is withheld as a spherical holdout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

from .calibration import Landmark, examine_fit
from .contracts import Problem, number
from .projection import (LEAN_LIMIT, RasterHoldout, _radial, _theta, angular_distance,
                         camera_to_world)


AGREEMENT_STEPS = 360
HOLDOUT_COLUMNS = 180
HOLDOUT_ROWS = 45
AGREEMENT_COLORS = np.array(
    [(38, 48, 56), (246, 147, 152), (144, 224, 175), (254, 203, 125)], np.uint8)
AGREEMENT_NAMES = ("outside both", "disputed belt", "concordant", "one placement only")

if TYPE_CHECKING:
    from .camera import Camera
    from .projection import Lens


def lean_marks(raw: list) -> tuple[Landmark, ...]:
    if not isinstance(raw, list) or not 2 <= len(raw) <= 30:
        raise Problem("Leveling needs 2 to 30 surveyed corners.")
    return tuple(Landmark.parse(entry, index) for index, entry in enumerate(raw))


def _polar(tx: float, ty: float) -> tuple[float, float]:
    return float(np.hypot(tx, ty)), float(np.degrees(np.arctan2(ty, tx)) % 360)


def _parameters(vector: np.ndarray, template: "Lens", mode: str) -> "Lens":
    from .projection import Lens

    if mode == "fixed_circle":
        tx, ty, north = vector
        center, radius = template.center_uv, template.radius_over_short_side
    else:
        cx, cy, radius, tx, ty, north = vector
        center = (float(cx), float(cy))
    tilt, bearing = _polar(float(tx), float(ty))
    return Lens(center, float(radius), template.maximum_zenith, float(north) % 360,
                template.model, (tilt, bearing), None)


def _residuals(vector, template, camera, pixels, world, weights, mode):
    trial = _parameters(vector, template, mode)
    projected, _ = trial.project_pixels(world, camera)
    return ((projected - pixels) * weights[:, None]).reshape(-1)


def _initial_north(template: "Lens", camera: "Camera", pixels: np.ndarray, world: np.ndarray) -> float:
    from .projection import bearing_delta

    offset = pixels - camera.unit_to_pixel(template.center_uv)
    lengths = np.linalg.norm(offset, axis=1)
    apparent = np.degrees(np.arctan2(offset[:, 0], -offset[:, 1]))
    deltas = np.radians(bearing_delta(apparent, world[:, 0] + template.north_clockwise))
    return template.north_clockwise + float(np.degrees(np.arctan2(
        np.sum(lengths * np.sin(deltas)), np.sum(lengths * np.cos(deltas)))))


def lean_fit(camera: "Camera", template: "Lens", landmarks: tuple[Landmark, ...], mode: str):
    if mode not in ("fixed_circle", "adjusted_circle"):
        raise Problem("Choose a fixed-circle or adjusted-circle placement.")
    pixels = camera.unit_to_pixel([mark.image for mark in landmarks])
    world = np.array([mark.direction for mark in landmarks], dtype=float)
    weights = np.sqrt(np.array([mark.confidence for mark in landmarks]))
    shifted = pixels - pixels.mean(axis=0)
    spans = np.linalg.svd(shifted, compute_uv=False)
    if len(landmarks) > 2 and (spans[1] < max(camera.width, camera.height) * .004
                               or spans[1] * 8 < spans[0]):
        raise Problem("Spread the marked corners in two image directions before leveling.")
    if mode == "adjusted_circle" and len(landmarks) < 3:
        raise Problem("Adjusting the circle needs at least three corners; use the fixed circle with two.")
    if np.any((90 - world[:, 1]) > template.maximum_zenith):
        raise Problem("Some surveyed corners fall beyond the indicated lens radius.")
    if np.any(world[:, 1] >= 89.99):
        raise Problem("Replace any near-zenith corner with one that has a definite bearing.")
    north0 = _initial_north(template, camera, pixels, world)
    bearing_span = abs(float((world[:, 0].max() - world[:, 0].min() + 180) % 360 - 180))
    warnings = []
    if len(landmarks) < 4 or bearing_span < 90:
        warnings.append("Corners cover only one side; keep the disputed belt withheld until more sides are surveyed.")
    if mode == "fixed_circle":
        x0 = np.array([0.0, 0.0, north0])
        lower = [-LEAN_LIMIT, -LEAN_LIMIT, north0 - 180]
        upper = [LEAN_LIMIT, LEAN_LIMIT, north0 + 180]
        scale = [3.0, 3.0, 110.0]
    else:
        x0 = np.array([*template.center_uv, template.radius_over_short_side, 0.0, 0.0, north0])
        lower = [0, 0, .065, -LEAN_LIMIT, -LEAN_LIMIT, north0 - 180]
        upper = [1, 1, 2.5, LEAN_LIMIT, LEAN_LIMIT, north0 + 180]
        scale = [.3, .3, .35, 3.0, 3.0, 110.0]
    result = least_squares(_residuals, np.clip(x0, lower, upper), bounds=(lower, upper),
                           args=(template, camera, pixels, world, weights, mode),
                           x_scale=scale, f_scale=max(2, min(camera.width, camera.height) / 215),
                           loss="soft_l1", max_nfev=300)
    if not result.success:
        raise Problem("Could not settle on a stable leveled placement. Check the corner marks.")
    if np.linalg.matrix_rank(result.jac, tol=.05) < len(x0):
        raise Problem("The surveyed corners do not constrain every placement setting.")
    lens = _parameters(result.x, template, mode)
    if lens.lean[0] > LEAN_LIMIT - 1e-6:
        raise Problem("The recovered tilt reaches the supported limit; re-survey the corners or check the lens.")
    fit = examine_fit(camera, template, lens, landmarks)
    return lens, fit, tuple(warnings)


def _inverse_directions(lens: "Lens", camera: "Camera", points: np.ndarray):
    center = camera.unit_to_pixel(lens.center_uv)
    pixel = camera.unit_to_pixel(points)
    dx, dy = np.moveaxis(pixel - center, -1, 0)
    fraction = np.hypot(dx, dy) / lens.scale_px(camera)
    d = fraction * _radial(np.radians(lens.maximum_zenith), lens.model)
    theta, possible = _theta(d, lens.model)
    camera_vectors = np.stack((np.sin(theta) * np.sin(np.arctan2(dx, -dy)),
                               np.sin(theta) * np.cos(np.arctan2(dx, -dy)),
                               np.cos(theta)), axis=-1)
    x, y, z = np.moveaxis(camera_to_world(camera_vectors, lens.lean), -1, 0)
    bearing = np.degrees(np.arctan2(x, y)) % 360
    elevation = np.degrees(np.arctan2(z, np.hypot(x, y)))
    inside = possible & (fraction <= 1 + 1e-7) & (elevation >= -1e-7)
    return np.stack((bearing, elevation), axis=-1), inside


def _native_state(lens_a: "Lens", lens_b: "Lens", camera: "Camera", threshold: float) -> np.ndarray:
    gx, gy = np.meshgrid(np.linspace(0, 1, AGREEMENT_STEPS), np.linspace(0, 1, AGREEMENT_STEPS))
    points = np.stack((gx, gy), axis=-1)
    dir_a, inside_a = _inverse_directions(lens_a, camera, points)
    dir_b, inside_b = _inverse_directions(lens_b, camera, points)
    disagreement = angular_distance(dir_a, dir_b)
    both = inside_a & inside_b
    state = np.zeros(points.shape[:2], dtype=np.uint8)
    state[inside_a | inside_b] = 3
    state[both & (disagreement > threshold)] = 1
    state[both & (disagreement <= threshold)] = 2
    return state


def _angular_states(lens_a: "Lens", lens_b: "Lens", camera: "Camera",
                    native: np.ndarray) -> np.ndarray:
    centers = np.stack(np.meshgrid(
        (np.arange(HOLDOUT_COLUMNS) + .5) * 360 / HOLDOUT_COLUMNS,
        (np.arange(HOLDOUT_ROWS) + .5) * 90 / HOLDOUT_ROWS), axis=-1)
    geo_a = lens_a.world_to_image(centers, camera)[1]
    geo_b = lens_b.world_to_image(centers, camera)[1]
    sub = np.linspace(.25, .75, 3)
    offsets = np.stack(np.meshgrid(sub, sub), axis=-1).reshape(-1, 2) - .5
    points = centers[:, :, None, :] + offsets * np.array([360 / HOLDOUT_COLUMNS, 90 / HOLDOUT_ROWS])
    pa, _ = lens_a.world_to_image(points, camera)
    clash = np.zeros(centers.shape[:2], dtype=bool)
    row = np.clip((pa[..., 1] * (AGREEMENT_STEPS - 1)).round().astype(int), 0, AGREEMENT_STEPS - 1)
    col = np.clip((pa[..., 0] * (AGREEMENT_STEPS - 1)).round().astype(int), 0, AGREEMENT_STEPS - 1)
    for k in range(offsets.shape[0]):
        clash |= native[row[..., k], col[..., k]] == 1
    states = np.zeros(centers.shape[:2], dtype=np.uint8)
    states[geo_a | geo_b] = 3
    states[geo_a & geo_b & clash] = 1
    states[geo_a & geo_b & ~clash] = 2
    return states


def _sector_rows(states: np.ndarray) -> list[dict]:
    bands = 12
    rows = []
    total = states.size
    for band in range(bands):
        part = states[:, band * HOLDOUT_COLUMNS // bands:(band + 1) * HOLDOUT_COLUMNS // bands]
        counts = np.bincount(part.reshape(-1), minlength=4)
        rows.append({
            "band": band + 1,
            "bearing_start": number(band * 360 / bands, 4),
            "bearing_end": number((band + 1) * 360 / bands, 4),
            "concordant_percent": number(100 * counts[2] / total),
            "disputed_percent": number(100 * counts[1] / total),
            "single_placement_percent": number(100 * counts[3] / total),
            "outside_both_percent": number(100 * counts[0] / total),
        })
    return rows


def _fit_entry(mode: str, lens: "Lens", fit, fit_warnings: tuple[str, ...]) -> dict:
    proof = fit.public()
    proof["mode"] = mode
    proof["warnings"] = list(fit_warnings)
    return {"mode": mode, "lens": lens.public(), "fit": proof, "warnings": list(fit_warnings)}


def _agreement_grid(states: np.ndarray):
    from .contracts import Footprint

    class _AgreementGrid:
        footprint = Footprint(0.0, 360.0, 0.0, 90.0)
        sine_edges = np.linspace(0, 1, states.shape[0] + 1)
        bearing_edges = np.linspace(0, 360, states.shape[1] + 1)

        def __init__(self, layer):
            self.layer = layer

        @property
        def shape(self):
            return self.layer.shape

    return _AgreementGrid(states)


def leveling_agreement(camera: "Camera", fixed: "Lens", adjusted: "Lens",
                       fixed_fit, adjusted_fit, fixed_warnings: tuple[str, ...],
                       adjusted_warnings: tuple[str, ...], threshold: float) -> dict:
    from .visual import GOLD, MINT, PINK, angular_map, calibration_view, comparison_panel, photo_canvas

    native = _native_state(fixed, adjusted, camera, threshold)
    states = _angular_states(fixed, adjusted, camera, native)
    disputed_holdout = RasterHoldout.from_mask(states == 1)
    cautious_holdout = RasterHoldout.from_mask(states != 2)
    counts = np.bincount(states.reshape(-1), minlength=4)
    summary = {
        "disagreement_threshold_deg": number(threshold),
        "concordant_cells": int(counts[2]), "disputed_cells": int(counts[1]),
        "single_placement_cells": int(counts[3]), "outside_both_cells": int(counts[0]),
        "concordant_percent": number(100 * counts[2] / states.size),
        "disputed_percent": number(100 * counts[1] / states.size),
        "single_placement_percent": number(100 * counts[3] / states.size),
        "holdout_grid": [HOLDOUT_COLUMNS, HOLDOUT_ROWS],
        "disputed_holdout_cells": disputed_holdout.cell_count(),
        "cautious_holdout_cells": cautious_holdout.cell_count(),
    }
    tint = AGREEMENT_COLORS[native]
    canvas = photo_canvas(camera, 880)
    from PIL import Image

    tint_layer = Image.fromarray(tint).resize(canvas.size, Image.Resampling.NEAREST)
    base = np.asarray(canvas)
    native_image = np.rint(base * .55 + np.asarray(tint_layer) * .45).astype(np.uint8)
    angular = angular_map(
        AGREEMENT_COLORS[states], _agreement_grid(states),
        "Where both leveled placements stand",
        {AGREEMENT_NAMES[2]: MINT, AGREEMENT_NAMES[1]: PINK,
         AGREEMENT_NAMES[3]: GOLD, AGREEMENT_NAMES[0]: (96, 112, 122)})
    panel = comparison_panel([
        ("Fixed measured circle", calibration_view(camera, fixed)),
        ("Circle adjusted to corners", calibration_view(camera, adjusted)),
    ], "Azimuth and elevation lines under both placements")
    return {
        "fits": [_fit_entry("fixed_circle", fixed, fixed_fit, fixed_warnings),
                 _fit_entry("adjusted_circle", adjusted, adjusted_fit, adjusted_warnings)],
        "summary": summary,
        "sectors": _sector_rows(states),
        "holdouts": {"disputed": disputed_holdout.public(), "cautious": cautious_holdout.public()},
        "images": {"native": native_image, "angular": angular, "panel": panel},
    }


def lens_with_holdout(lens: "Lens", holdout: RasterHoldout | None) -> "Lens":
    from .projection import Lens

    return Lens(lens.center_uv, lens.radius_over_short_side, lens.maximum_zenith,
                lens.north_clockwise, lens.model, lens.lean, holdout)


def single_placement(camera: "Camera", fixed: "Lens", fixed_fit,
                     fixed_warnings: tuple[str, ...], reason: str) -> dict:
    """Fallback when only the fixed-circle placement has enough corners."""
    from .visual import calibration_view

    warnings = list(fixed_warnings) + [reason]
    entry = _fit_entry("fixed_circle", fixed, fixed_fit, tuple(warnings))
    panel = calibration_view(camera, fixed)
    empty = RasterHoldout.from_mask(np.zeros((HOLDOUT_ROWS, HOLDOUT_COLUMNS), bool))
    return {
        "fits": [entry],
        "summary": {"note": "Only the fixed-circle placement is available; add a third corner to compare both.",
                    "reason": reason, "disagreement_threshold_deg": None},
        "sectors": [],
        "holdouts": {"disputed": empty.public(), "cautious": empty.public()},
        "images": {"native": None, "angular": None, "panel": panel},
    }
