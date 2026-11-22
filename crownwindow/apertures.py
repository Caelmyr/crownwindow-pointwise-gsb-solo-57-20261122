"""Connected open patches in an equal-solid-angle map."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .bench import Bench
from .contracts import boolean, fields, number, real
from .projection import Grid, directions_to_vectors, vectors_to_directions
from .result import Result
from .sky import MISSING, OPEN, UNCERTAIN
from .visual import angular_map, border_color_at


class _Linkage:
    def __init__(self, n: int):
        self.parents = np.arange(n + 1)

    def root(self, i: int) -> int:
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]
            i = int(self.parents[i])
        return i

    def join(self, a: int, b: int) -> None:
        first, second = self.root(a), self.root(b)
        if first != second:
            self.parents[max(first, second)] = min(first, second)

    def lookup(self) -> np.ndarray:
        return np.array([self.root(i) for i in range(len(self.parents))])


def _connected(known_open: np.ndarray, periodic: bool, diagonal: bool) -> np.ndarray:
    neighborhood = ndimage.generate_binary_structure(2, 2 if diagonal else 1)
    original, n = ndimage.label(known_open, structure=neighborhood)
    if not periodic or not n:
        return original
    links = _Linkage(n)
    for row, a in enumerate(original[:, 0]):
        if not a:
            continue
        neighbours = range(max(row - 1, 0), min(row + 2, len(original))) if diagonal else (row,)
        for q in neighbours:
            b = original[q, -1]
            if b:
                links.join(int(a), int(b))
    return links.lookup()[original]


def _minimum_bearing_span(bearings: np.ndarray, step: float, periodic: bool) -> tuple[float, float]:
    if not len(bearings):
        return 0, 0
    if not periodic:
        clipped = np.sort(bearings)
        return float(clipped[0]) % 360, min(360., float(clipped[-1] - clipped[0] + step))
    positions = np.unique(np.asarray(bearings) % 360)
    gaps = np.diff(np.r_[positions, positions[0] + 360])
    end = int(np.argmax(gaps))
    return float((positions[(end + 1) % len(positions)] - step / 2) % 360), min(360., float(360 - gaps[end] + step))


def _unknown_rim(labels: np.ndarray, classes: np.ndarray, periodic: bool) -> dict[int, int]:
    if periodic:
        padded = np.pad(classes, ((1, 1), (0, 0)), mode="constant", constant_values=MISSING)
        padded = np.pad(padded, ((0, 0), (1, 1)), mode="wrap")
    else:
        padded = np.pad(classes, 1, mode="constant", constant_values=MISSING)
    has_uncertain_contact = np.zeros(labels.shape, dtype=bool)
    for dy, dx in ((0, 1), (1, 0), (1, 2), (2, 1)):
        layer = padded[dy:dy + labels.shape[0], dx:dx + labels.shape[1]]
        has_uncertain_contact |= (layer == UNCERTAIN) | (layer == MISSING)
    counts = np.bincount(labels[has_uncertain_contact], minlength=int(labels.max()) + 1)
    return {int(index): int(count) for index, count in enumerate(counts) if index}


def _shape_rows(labels: np.ndarray, grid: Grid, classes: np.ndarray, min_sr: float) -> tuple[list[dict], np.ndarray]:
    populations = np.bincount(labels.reshape(-1))
    ordered = sorted((i for i in range(1, len(populations)) if populations[i] * grid.solid_angle_per_cell >= min_sr),
                     key=lambda i: (-populations[i], i))
    rim = _unknown_rim(labels, classes, grid.periodic)
    rows = []
    remap = np.zeros(len(populations), dtype=np.int32)
    for rank, old in enumerate(ordered[:160], 1):
        selected = labels == old
        directions = grid.directions[selected]
        vectors = directions_to_vectors(directions)
        central = vectors.mean(axis=0)
        magnitude = float(np.linalg.norm(central))
        direction = vectors_to_directions(central) if magnitude >= 1e-4 else None
        elevation = directions[:, 1]
        first, sweep = _minimum_bearing_span(
            grid.bearing_edges[:-1][np.any(selected, axis=0)],
            grid.footprint.sweep / grid.shape[1], grid.periodic)
        rows.append({
            "window": rank,
            "center_direction": [number(v, 4) for v in direction] if direction is not None else None,
            "mean_direction_concentration": number(magnitude, 6),
            "solid_angle_sr": number(populations[old] * grid.solid_angle_per_cell, 7),
            "measured_pixels_in_angular_grid": int(populations[old]),
            "azimuth_start": number(first, 4),
            "azimuth_coverage": number(sweep, 4),
            "elevation_of_centers": [number(v, 4) for v in (elevation.min(), elevation.max())],
            "touching_unknown_cells": rim.get(old, 0),
        })
        remap[old] = rank
    return rows, remap[labels]


def _colorise(grid: Grid, classes: np.ndarray, windows: np.ndarray):
    palette = np.array([(49, 55, 62), (51, 77, 62), (97, 123, 132), (177, 130, 68)], np.uint8)
    result = palette[classes]
    for item in range(1, int(windows.max()) + 1):
        result[windows == item] = border_color_at(item)
    return angular_map(result, grid, "Observed canopy openings",
                       {"Each retained window": (148, 188, 237), "Ambiguous boundary": (177, 130, 68),
                        "No camera evidence": (49, 55, 62)})


def windows(bench: Bench) -> Result:
    source = bench.one()
    options = bench.order.settings
    fields(options, "diagonal", "minimum")
    diagonal = boolean(options.get("diagonal", False), "Connect corners")
    minimum = real(options.get("minimum", .002), "Minimum opening solid angle", .00001, 2)
    grid = bench.grid
    sample = source.sample(grid)
    labels = _connected(sample.codes == OPEN, grid.periodic, diagonal)
    rows, map_data = _shape_rows(labels, grid, sample.codes, minimum)
    kept = len(rows)
    result = Result("sky-windows", "Observed gaps in the canopy",
                    "An opening is a connected patch of decisive sky. Amber and unmeasured cells remain outside its reported area.")
    result.metrics = {
        "listed_openings": kept,
        "minimum_solid_angle_sr": minimum,
        "periodic_north_boundary": grid.periodic,
        "connect_touching_corners": diagonal,
        "listed_opening_solid_angle_sr": number(sum(row["solid_angle_sr"] for row in rows), 7),
        "open_but_unlisted_cells": int(np.count_nonzero((sample.codes == OPEN) & (map_data == 0))),
        "sampling": grid.describe(),
    }
    result.readings = rows
    result.png("sky-windows", "Connected decisive sky", _colorise(grid, sample.codes, map_data))
    result.csv("opening-register", "Gap geometry table", rows,
               ["window", "center_direction", "mean_direction_concentration", "solid_angle_sr",
                "measured_pixels_in_angular_grid", "azimuth_start", "azimuth_coverage", "elevation_of_centers",
                "touching_unknown_cells"])
    result.warnings = source.method_warnings()
    result.probe_kind = "angular-cell"
    return result
