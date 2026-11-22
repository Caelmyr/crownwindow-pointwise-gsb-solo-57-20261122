"""Comparative hemispheric luminance and two-observation changes."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .bench import Bench
from .camera import rgb_luminance, srgb_to_linear
from .contracts import Problem, fields, number, real, whole
from .result import Result
from .sky import LABELS, MISSING, OCCLUDED, OPEN, UNCERTAIN
from .visual import angular_map, category_map, comparison_panel, normalized_heat, toned_raw


def _weighted_bounds(codes: np.ndarray, luma: np.ndarray, weight: np.ndarray) -> dict:
    known_sky = codes == OPEN
    uncertain = (codes == UNCERTAIN) | (codes == MISSING)
    seen = (codes != MISSING)
    contribution = np.where(known_sky, luma * weight, 0)
    known_light = float(np.sum(contribution))
    total_weight = float(np.sum(weight))
    unknown_weight = float(np.sum(weight[uncertain]))
    sky_weight = float(np.sum(weight[known_sky]))
    return {
        "known_open_luminance_weight": number(known_light, 7),
        "possible_additional_unresolved_weight": number(unknown_weight, 7),
        "potential_weight_normalization": number(total_weight, 7),
        "weighted_measured_fraction": number(np.sum(weight[seen]) / total_weight),
        "weighted_open_fraction_minimum": number(sky_weight / total_weight),
        "relative_camera_index_bounds": [number(known_light / total_weight, 6),
                                           number((known_light + unknown_weight) / total_weight, 6)],
        "weighted_open_pixel_mean": number(known_light / sky_weight) if sky_weight else None,
    }


def _brightness_sectors(codes, luma, weight, directions, count):
    rows = []
    for index, keep in enumerate(np.array_split(np.arange(codes.shape[1]), count)):
        rendered = _weighted_bounds(codes[:, keep], luma[:, keep], weight[:, keep])
        rows.append({
            "sector": index + 1,
            "first_center_bearing": number(directions[0, keep[0], 0], 4),
            "last_center_bearing": number(directions[0, keep[-1], 0], 4),
            "open_weight": rendered["weighted_open_fraction_minimum"],
            "measured_weight": rendered["weighted_measured_fraction"],
            "relative_minimum": rendered["relative_camera_index_bounds"][0],
            "relative_maximum": rendered["relative_camera_index_bounds"][1],
            "open_luminance": rendered["weighted_open_pixel_mean"],
        })
    return rows


def diffuse_reference(bench: Bench) -> Result:
    source = bench.one()
    data = bench.order.settings
    fields(data, "sectors", "dark_level", "white_level")
    sectors = whole(data.get("sectors", 8), "Directional slices", 3, 36)
    dark = real(data.get("dark_level", 0), "Camera dark reference", 0, .95)
    white = real(data.get("white_level", 1), "Camera bright reference", .05, 1)
    if white - dark < .05:
        raise Problem("Choose camera reference endpoints at least .05 apart.")
    grid = bench.grid
    samples = source.sample(grid)
    raw = rgb_luminance(srgb_to_linear(samples.rgb))
    luma = np.clip((raw - dark) / (white - dark), 0, 1)
    weights = np.sin(np.radians(grid.directions[..., 1])) * grid.solid_angle_per_cell
    summary = _weighted_bounds(samples.codes, luma, weights)
    rows = _brightness_sectors(samples.codes, luma, weights, grid.directions, sectors)
    observed = samples.codes == OPEN
    summary["confirmed_open_cells"] = int(observed.sum())
    report = Result("diffuse-light", "Diffuse sky reference",
                    "An uncalibrated camera-light index weights visible sky by its angle to a horizontal surface.")
    report.metrics = {**summary, "dark_reference": dark, "white_reference": white, "sampling": grid.describe()}
    report.readings = rows
    contribution = np.sin(np.radians(grid.directions[..., 1])) * luma
    report.png("weighted-sky", "Measured sky contribution", angular_map(
        normalized_heat(contribution, observed, 0, 1), grid, "Relative sky brightness × elevation weight"))
    report.png("camera-luminance", "Open sky camera luminance", angular_map(
        normalized_heat(luma, observed, 0, 1), grid, "Measured sky luminance after reference scaling"))
    report.png("sky-basis", "Visible basis and gaps", category_map(samples.codes, grid, "Evidence states"))
    report.csv("sector-brightness", "Relative sky brightness by bearing", rows)
    report.warnings = [*source.method_warnings(),
                       "Camera code is not a calibrated radiance measurement. Do not compare different exposures as absolute energy."]
    report.probe_kind = "direction"
    return report


CHANGE_CLASSES = (
    "both_missing", "one_missing", "one_uncertain", "unchanged_obstructed",
    "unchanged_open", "new_open", "new_obstruction",
)
CHANGE_PALETTE = np.array([(53, 59, 67), (109, 104, 125), (211, 159, 80), (37, 80, 70),
                           (143, 185, 195), (128, 232, 179), (244, 130, 137)], dtype=np.uint8)


def classify_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    both_missing = (a == MISSING) & (b == MISSING)
    one_missing = ((a == MISSING) | (b == MISSING)) & ~both_missing
    settled = ~one_missing & ~both_missing & (a != UNCERTAIN) & (b != UNCERTAIN)
    result = np.full(a.shape, 2, np.uint8)
    result[both_missing] = 0
    result[one_missing] = 1
    result[settled & (a == OCCLUDED) & (b == OCCLUDED)] = 3
    result[settled & (a == OPEN) & (b == OPEN)] = 4
    result[settled & (a == OCCLUDED) & (b == OPEN)] = 5
    result[settled & (a == OPEN) & (b == OCCLUDED)] = 6
    return result


def _changes(codes: np.ndarray, weight: float) -> list[dict]:
    counts = np.bincount(codes.flatten(), minlength=len(CHANGE_CLASSES))
    return [{"state": state, "cells": int(count), "area_sr": number(count * weight, 8),
             "percent_of_footprint": number(count / codes.size * 100)}
            for state, count in zip(CHANGE_CLASSES, counts, strict=True)]


def _difference(rgb_a, rgb_b, jointly, floor):
    luma_a = rgb_luminance(srgb_to_linear(rgb_a))
    luma_b = rgb_luminance(srgb_to_linear(rgb_b))
    signed = luma_b - luma_a
    rendered = np.empty(rgb_a.shape, dtype=np.uint8)
    magnitude = np.clip(np.abs(signed) / max(floor * 5, .01), 0, 1)
    base = np.array([65, 67, 76], dtype=float)
    positive = np.array([122, 214, 163])
    negative = np.array([250, 125, 143])
    tone = np.where((signed >= 0)[..., None], positive, negative)
    rendered[:] = base + (tone - base) * magnitude[..., None]
    rendered[~jointly] = [38, 44, 55]
    return signed, rendered


def _change_landmarks(grid, codes: np.ndarray, classes: tuple[int, ...]) -> list[dict]:
    """A few largest contiguous changes for revisiting the same bearing."""
    rows = []
    for state in classes:
        labels, largest = ndimage.label(codes == state)
        counts = np.bincount(labels.flatten(), minlength=largest + 1)
        order = sorted(range(1, largest + 1), key=lambda key: -counts[key])
        for rank, component in enumerate(order[:10], 1):
            uv = np.argwhere(labels == component)
            if len(uv) < 3:
                continue
            distances = ((uv - uv.mean(axis=0)) ** 2).sum(axis=1)
            representative = uv[np.argmin(distances)]
            i, j = representative
            bearing, elevation = grid.directions[i, j]
            rows.append({"state": CHANGE_CLASSES[state], "rank": rank, "cells": len(uv),
                         "bearing": number(bearing, 5), "elevation": number(elevation, 5),
                         "solid_angle_sr": number(counts[component] * grid.solid_angle_per_cell, 7)})
    return rows


def refind_change(bench: Bench) -> Result:
    before, after = bench.two()
    data = bench.order.settings
    fields(data, "difference_floor")
    floor = real(data.get("difference_floor", .08), "Luminance change indicator", .005, .60)
    grid = bench.grid
    earlier, later = before.sample(grid), after.sample(grid)
    both_seen = earlier.measured & later.measured
    labels = classify_pair(earlier.codes, later.codes)
    evidence, differences = _difference(earlier.rgb, later.rgb, both_seen, floor)
    rows = _changes(labels, grid.solid_angle_per_cell)
    changed = both_seen & (np.abs(evidence) >= floor)
    report = Result("paired-change", "Same direction on repeated observations",
                    "Openings and obstructions change only where both camera views establish a class.")
    areas = {row["state"]: row["area_sr"] for row in rows}
    measured = int(np.count_nonzero(both_seen))
    report.metrics = {"classes": rows, "overlapping_camera_cells": measured,
                      "open_area_gained_sr": areas["new_open"], "open_area_lost_sr": areas["new_obstruction"],
                      "brightness_change_fraction_in_overlap": number(changed.sum() / measured) if measured else None,
                      "difference_floor": floor, "sampling": grid.describe()}
    report.readings = _change_landmarks(grid, labels, (5, 6))
    colors = {"New opening": CHANGE_PALETTE[5], "New obstruction": CHANGE_PALETTE[6],
              "Uncertain class": CHANGE_PALETTE[2], "One camera missing": CHANGE_PALETTE[1]}
    report.png("observed-change", "Confirmed and unresolved angular comparison",
               angular_map(CHANGE_PALETTE[labels], grid, "Repeat survey in common directions", colors))
    report.png("comparison-basis", "Both surveys in the same angular coordinates",
               comparison_panel([(before.title, angular_map(toned_raw(earlier.rgb, earlier.measured), grid, "Earlier")),
                                 (after.title, angular_map(toned_raw(later.rgb, later.measured), grid, "Later"))],
                                "Common bearing and elevation coordinate"))
    report.png("brightness-difference", "Overlapping luminance difference",
               angular_map(differences, grid, "Later minus earlier camera luminance",
                           {"Brighter": (122, 214, 163), "Darker": (250, 125, 143), "Unmeasured": (38, 44, 55)}))
    report.csv("change-areas", "Area by classification change", rows)
    report.csv("revisit-directions", "Larger changes to revisit", report.readings,
               ["state", "rank", "cells", "bearing", "elevation", "solid_angle_sr"])
    report.warnings = [*before.method_warnings(), *after.method_warnings(),
                       "Treat color differences as camera observations; exposure and weather can change them."]
    report.probe_kind = "direction"
    return report
