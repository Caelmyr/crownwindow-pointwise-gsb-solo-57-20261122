"""Directional view factors for small oriented surfaces at the station."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bench import Bench
from .camera import rgb_luminance, srgb_to_linear
from .contracts import Problem, angle, fields, items, mapping, number, real, text
from .projection import directions_to_vectors
from .result import Result
from .sky import MISSING, OCCLUDED, OPEN, UNCERTAIN
from .visual import GOLD, MINT, PINK, angular_map, comparison_panel, height_chart, normalized_heat


@dataclass(frozen=True)
class Plane:
    name: str
    azimuth: float
    normal_altitude: float

    @classmethod
    def parse(cls, raw: dict, index: int) -> Plane:
        data = mapping(raw, "Receiving plane")
        fields(data, "name", "normal")
        bearing, elevation = angle(data.get("normal"), "Surface normal")
        name = text(data.get("name") or f"Surface {index + 1}", "Surface name", 54)
        return cls(name, bearing, elevation)

    def incidence(self, directions: np.ndarray) -> np.ndarray:
        normal = directions_to_vectors(np.array([self.azimuth, self.normal_altitude]))
        vector = directions_to_vectors(directions)
        return np.maximum(0, vector @ normal)


DEMO_PLANES = [{"name": "Horizontal plot marker", "normal": [0, 90]},
               {"name": "East-facing surface", "normal": [90, 25]},
               {"name": "West-facing surface", "normal": [270, 25]}]


def _readable_plane_row(plane: Plane, matrix: np.ndarray, codes: np.ndarray, luminance: np.ndarray,
                        cell_area: float):
    incidence = matrix * cell_area
    physical = (incidence > 0)
    possible = float(incidence.sum())
    area_seen = float(incidence[codes != MISSING].sum())
    certain_open = float(incidence[codes == OPEN].sum())
    certain_blocked = float(incidence[codes == OCCLUDED].sum())
    unresolved = possible - certain_open - certain_blocked
    day_score = float((luminance * incidence)[codes == OPEN].sum())
    reported = {
        "plane": plane.name,
        "normal_bearing": plane.azimuth,
        "normal_elevation": plane.normal_altitude,
        "incident_cells": int(physical.sum()),
        "effective_sector_weight": number(possible, 8),
        "camera_support_weight": number(area_seen, 8),
        "minimum_open_weight": number(certain_open, 8),
        "confirmed_blocked_weight": number(certain_blocked, 8),
        "unknown_weight": number(max(0, unresolved), 8),
        "open_fraction_lower": number(certain_open / possible) if possible > 1e-8 else None,
        "open_fraction_upper": number((certain_open + max(0, unresolved)) / possible) if possible > 1e-8 else None,
        "relative_light_index": number(day_score / possible) if possible > 1e-8 else None,
    }
    return reported


def _dominant_sectors(plane: Plane, kernel: np.ndarray, directions: np.ndarray, codes: np.ndarray, north: float):
    angle_offset = (directions[..., 0] - north) % 360
    cells = np.floor(angle_offset / 45).astype(int)
    parts = []
    for index in range(8):
        interval = cells == index
        supported = interval & (kernel > 0)
        if not supported.any():
            continue
        values = kernel[supported]
        visible = codes[supported]
        sum_all = values.sum()
        parts.append({
            "plane": plane.name, "start_bearing": number((north + 45 * index) % 360),
            "end_bearing": number((north + 45 * (index + 1)) % 360),
            "has_incident_rays": int(len(values)),
            "lower_open_share": number(values[visible == OPEN].sum() / sum_all),
            "upper_open_share": number(values[visible != OCCLUDED].sum() / sum_all),
            "unobserved_share": number(values[visible == MISSING].sum() / sum_all),
            "ambiguous_share": number(values[visible == UNCERTAIN].sum() / sum_all),
        })
    return parts


def oriented_surfaces(bench: Bench) -> Result:
    source = bench.one()
    data = bench.order.settings
    fields(data, "surfaces", "indicated_bearing")
    raw = items(data.get("surfaces", DEMO_PLANES), "Receiving surfaces", 1, 6)
    planes = [Plane.parse(value, index) for index, value in enumerate(raw)]
    axis = real(data.get("indicated_bearing", 0), "Start of direction sectors", -360, 720)
    if len({plane.name for plane in planes}) != len(planes):
        raise Problem("Give each receiving surface a distinct label.")
    grid = bench.grid
    samples = source.sample(grid)
    luma = rgb_luminance(srgb_to_linear(samples.rgb))
    rows = []
    maps = []
    detail = []
    for plane in planes:
        kernel = plane.incidence(samples.directions)
        row = _readable_plane_row(plane, kernel, samples.codes, luma, grid.solid_angle_per_cell)
        rows.append(row)
        detail.extend(_dominant_sectors(plane, kernel, samples.directions, samples.codes, axis))
        if len(maps) < 3:
            rgb = normalized_heat(kernel, (samples.codes == OPEN) & (kernel > 0))
            maps.append((plane.name, angular_map(rgb, grid, "Incident open-sky support")))
    report = Result("surface-orientation", "Sky exposure by surface orientation",
                    "Compare the open-view contribution to small surfaces angled differently at the same point.")
    report.metrics = {"surfaces": len(rows), "common_observation": source.title,
                      "surfaces_with_no_modelled_incidence": sum(row["open_fraction_lower"] is None for row in rows),
                      "sampling": grid.describe()}
    report.readings = rows
    report.png("oriented-support", "Directional contribution by surface", comparison_panel(maps, "Oriented surface sky view"))
    report.png("oriented-range", "Possible fractions by orientation", height_chart([
        {"name": "Confirmed sky", "color": MINT, "values": [r["open_fraction_lower"] for r in rows]},
        {"name": "Possible sky", "color": GOLD, "values": [r["open_fraction_upper"] for r in rows]},
        {"name": "Observed brightness index", "color": PINK, "values": [r["relative_light_index"] for r in rows]},
    ], "Orientation and aperture bounds", "Fraction of selected view", "Surfaces in entered order", hard_bounds=(0, 1)))
    report.csv("surface-factors", "Envelope for each surface", rows)
    report.csv("surface-sector-contributions", "Contributions separated by bearing", detail)
    report.warnings = [*source.method_warnings(),
                       "The calculation describes only the selected sky footprint and receiving surface direction; no terrain radiance is included."]
    report.probe_kind = "direction"
    return report
