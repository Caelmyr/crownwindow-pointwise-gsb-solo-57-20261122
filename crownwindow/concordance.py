"""Evaluate field control points across two repeat photographs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bench import Bench, Source
from .contracts import Problem, fields, items, mapping, number, point, real, text
from .projection import angular_distance, directions_to_vectors, vectors_to_directions
from .result import Result
from .sky import MISSING, OCCLUDED, OPEN
from .visual import GOLD, MINT, PINK, comparison_panel, height_chart, plot_over_camera


@dataclass(frozen=True)
class TiePoint:
    label: str
    first: tuple[float, float]
    second: tuple[float, float]
    support: float

    @classmethod
    def parse(cls, raw: dict, index: int) -> TiePoint:
        data = mapping(raw, "Repeat field point")
        fields(data, "label", "first", "second", "support")
        return cls(
            text(data.get("label") or f"Tie {index + 1}", "Field point", 65),
            point(data.get("first"), "First photograph point"),
            point(data.get("second"), "Second photograph point"),
            real(data.get("support", 1), "Confidence in correspondence", .05, 1),
        )


DEMO_TIES = [
    {"label": "Near zenith", "first": [.50, .505], "second": [.505, .497], "support": .9},
    {"label": "Canopy corner", "first": [.58, .48], "second": [.558, .458], "support": .7},
    {"label": "Unverified landmark", "first": [.51, .69], "second": [.61, .66], "support": .4},
]


def _inference(source: Source, shape):
    return source.lens.image_to_world(np.asarray(shape, dtype=float), source.camera)


def _agreed_direction(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return vectors_to_directions(directions_to_vectors(a) + directions_to_vectors(b))


def _dot_direction(source: Source, uv: tuple[float, float], radius: float) -> dict:
    data = source.lens.readout(source.camera, uv)
    nearby = source.camera.support_window(uv, radius)
    return {"native": list(uv), "px": data["pixel"], "az_el": data["direction"], "visible": data["valid"],
            "tone": data["display_luminance"] if data["valid"] else None,
            "valid_nearby": nearby["supported_pixels"] / nearby["expected_pixels"],
            "lens_version": source.retained_lens["id"] if source.retained_lens else None}


def _local_bearing_spread(source: Source, xy: tuple[float, float], pixel_uncertainty: float) -> float | None:
    center = source.camera.unit_to_pixel(xy)
    offsets = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float) * pixel_uncertainty
    sample = source.camera.pixel_to_unit(center + np.r_[np.zeros((1, 2)), offsets])
    directions, enabled = _inference(source, sample)
    if not np.all(enabled):
        return None
    angular = np.asarray(angular_distance(directions[0], directions[1:]))
    return float(angular.max())


def _reference_sample(source: Source, direction: np.ndarray) -> dict:
    evidence = source.sample(direction[None, :])
    row = evidence.readout(0)
    row["decisive"] = int(evidence.codes[0]) in (OPEN, OCCLUDED)
    return row


def _evaluate(pairs: list[TiePoint], first: Source, second: Source, tolerance: float,
              neighborhood: float) -> tuple[list[dict], dict]:
    rows = []
    for mark in pairs:
        a = _dot_direction(first, mark.first, neighborhood)
        b = _dot_direction(second, mark.second, neighborhood)
        is_observed = a["visible"] and b["visible"] and a["az_el"] is not None and b["az_el"] is not None
        if not is_observed:
            rows.append(_invalid_row(mark, a, b))
            continue
        a_direction, b_direction = np.asarray(a["az_el"]), np.asarray(b["az_el"])
        variance = float(angular_distance(a_direction, b_direction))
        spread_a = _local_bearing_spread(first, mark.first, neighborhood)
        spread_b = _local_bearing_spread(second, mark.second, neighborhood)
        lower, upper = _angular_bracket(variance, spread_a, spread_b)
        expected = _agreed_direction(a_direction, b_direction)
        one = _reference_sample(first, expected)
        two = _reference_sample(second, expected)
        if lower is None:
            status = "near_lens_edge"
        elif lower > tolerance:
            status = "separated"
        elif upper <= tolerance:
            status = "within_tolerance"
        else:
            status = "overlapping_uncertainty"
        rows.append({
            "field_point": mark.label, "first_native": list(mark.first), "second_native": list(mark.second),
            "first_direction": a["az_el"], "second_direction": b["az_el"], "observed": True,
            "angular_separation": number(variance, 5), "min_separation": number(lower, 5),
            "max_separation": number(upper, 5), "status": status, "concordance_weight": mark.support,
            "midpoint_direction": [number(v, 5) for v in expected],
            "first_midpoint_class": one["class"], "second_midpoint_class": two["class"],
            "minimum_local_pixel_support": number(min(a["valid_nearby"], b["valid_nearby"])),
        })
    return rows, _rms(rows)


def _angular_bracket(measurement, left, right):
    if left is None or right is None:
        return None, None
    combined = left + right
    return max(0, measurement - combined), measurement + combined


def _invalid_row(mark, first, second):
    return {"field_point": mark.label, "first_native": list(mark.first), "second_native": list(mark.second),
            "first_direction": first["az_el"], "second_direction": second["az_el"], "observed": False,
            "angular_separation": None, "min_separation": None, "max_separation": None,
            "status": "outside_observation", "concordance_weight": mark.support, "midpoint_direction": None,
            "first_midpoint_class": None, "second_midpoint_class": None,
            "minimum_local_pixel_support": number(min(first["valid_nearby"], second["valid_nearby"]))}


def _rms(rows: list[dict]) -> dict:
    supported = [row for row in rows if row["observed"]]
    if not supported:
        return {"observed_pairs": 0, "weighted_rms_degrees": None, "median_separation_degrees": None}
    weights = np.array([r["concordance_weight"] for r in supported])
    distance = np.array([r["angular_separation"] for r in supported])
    return {"observed_pairs": len(distance),
            "weighted_rms_degrees": number(np.sqrt(np.average(distance ** 2, weights=weights)), 6),
            "median_separation_degrees": number(np.median(distance), 6)}


def _pair_overlay(one: Source, two: Source, rows: list[dict]):
    colors = {"within_tolerance": MINT, "separated": PINK, "outside_observation": (141, 158, 166),
              "near_lens_edge": GOLD, "overlapping_uncertainty": GOLD}
    pictures = []
    for source, key in zip((one, two), ("first_native", "second_native"), strict=True):
        points = []
        for i, row in enumerate(rows, 1):
            u, v = row[key]
            points.append((u, v, str(i) + " " + row["field_point"][:22], colors[row["status"]]))
        pictures.append((source.title, plot_over_camera(source.camera, points)))
    return comparison_panel(pictures, "Repeat measurements in both originals")


def compare_ties(bench: Bench) -> Result:
    first, second = bench.two()
    data = bench.order.settings
    fields(data, "pairs", "tolerance", "neighborhood")
    pairs = [TiePoint.parse(raw, i) for i, raw in enumerate(items(data.get("pairs", DEMO_TIES), "Field points", 1, 40))]
    tolerance = real(data.get("tolerance", 3), "Tolerance (degrees)", .25, 40)
    radius = real(data.get("neighborhood", 4), "Marking radius (pixels)", 1, 18)
    rows, error = _evaluate(pairs, first, second, tolerance, radius)
    report = Result("tie-audit", "Repeat photographs agree at these marks",
                    "Read each marked point through its own camera lens and measure angular agreement without changing either lens.")
    statuses = {state: sum(x["status"] == state for x in rows) for state in
                ("within_tolerance", "separated", "outside_observation", "near_lens_edge", "overlapping_uncertainty")}
    report.metrics = {**error, "states": statuses, "neighborhood_px": radius,
                      "allowed_separation_degrees": tolerance}
    report.readings = rows
    report.png("repeat-marks", "Tie positions in both sources", _pair_overlay(first, second, rows))
    interval = height_chart([
        {"name": "Measured separation", "color": PINK, "values": [r["angular_separation"] for r in rows]},
        {"name": "Lower support bound", "color": MINT, "values": [r["min_separation"] for r in rows]},
        {"name": "Upper support bound", "color": GOLD, "values": [r["max_separation"] for r in rows]},
    ], "Repeat angle at each field point", "Degrees", "Field marks in entered order", reference=tolerance)
    report.png("tie-residuals", "Angular separation and support", interval)
    report.csv("tie-measurements", "Repeat field point readings", rows)
    report.warnings = ["The displayed bounds only account for the indicated pixel marking neighborhood. Lens model and field repeat uncertainty may remain."]
    report.probe_kind = "direction"
    return report
