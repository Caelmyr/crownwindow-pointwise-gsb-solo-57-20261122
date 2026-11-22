"""Combine two partly obstructed captures into one directional canopy view.

Both photographs come from the same fixed station with their own calibrated
lens.  Every equal-solid-angle cell is sampled through each camera and the two
readings are combined without double counting the directions both cameras saw.
Where the two decisive readings disagree (for example leaves that moved in the
wind) the cells are grouped into review patches and the operator chooses which
capture supplies that patch.  Every merged cell keeps the identity of its
adopted capture, so a later point probe can return to the adopted original and
the opposite capture can still be opened for cross-checking.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .apertures import _Linkage
from .bench import Bench
from .contracts import Problem, fields, identifier, items, mapping, number, whole
from .result import Result
from .sky import LABELS, MISSING, OCCLUDED, OPEN, UNCERTAIN
from .visual import GOLD, PALE, angular_map, category_map, _draw_font


# Provenance categories painted on the source-attribution map.
PROV_FIRST_ONLY = 1
PROV_SECOND_ONLY = 2
PROV_AGREED_FIRST = 3
PROV_AGREED_SECOND = 4
PROV_CONFLICT_FIRST = 5
PROV_CONFLICT_SECOND = 6
PROV_NOTHING = 0
PROV_PALETTE = np.array([
    (45, 54, 65),    # neither capture measured the cell
    (96, 160, 214),  # first capture only
    (150, 205, 138),  # second capture only
    (214, 190, 116),  # both agree, first adopted
    (196, 150, 210),  # both agree, second adopted
    (244, 167, 86),  # conflict, first chosen
    (240, 126, 138),  # conflict, second chosen
], dtype=np.uint8)
PROV_LEGEND = {
    "First capture only": tuple(PROV_PALETTE[PROV_FIRST_ONLY]),
    "Second capture only": tuple(PROV_PALETTE[PROV_SECOND_ONLY]),
    "Agree · first adopted": tuple(PROV_PALETTE[PROV_AGREED_FIRST]),
    "Agree · second adopted": tuple(PROV_PALETTE[PROV_AGREED_SECOND]),
    "Conflict · first chosen": tuple(PROV_PALETTE[PROV_CONFLICT_FIRST]),
    "Conflict · second chosen": tuple(PROV_PALETTE[PROV_CONFLICT_SECOND]),
    "Neither capture": tuple(PROV_PALETTE[PROV_NOTHING]),
}
DECISIVE = (OPEN, OCCLUDED)
MAX_CONFLICT_PATCHES = 120


def _connected_conflicts(mask: np.ndarray, periodic: bool) -> np.ndarray:
    """Label connected disagreement cells, joining across the north wrap."""
    structure = ndimage.generate_binary_structure(2, 2)
    labels, count = ndimage.label(mask, structure=structure)
    if not periodic or not count:
        return labels
    links = _Linkage(count)
    for row, first in enumerate(labels[:, 0]):
        if not first:
            continue
        for dy in (-1, 0, 1):
            other_row = row + dy
            if 0 <= other_row < labels.shape[0]:
                second = labels[other_row, -1]
                if second:
                    links.join(int(first), int(second))
    return links.lookup()[labels]


def _parse_choices(raw: object, first_id: str, second_id: str) -> dict[int, int]:
    """Map each reviewed patch ordinal to the chosen source index (0 or 1)."""
    entries = items(raw or [], "Conflict patch choices", 0, MAX_CONFLICT_PATCHES)
    choices: dict[int, int] = {}
    for entry in entries:
        data = mapping(entry, "Conflict patch choice")
        fields(data, "patch", "source")
        patch = whole(data.get("patch"), "Conflict patch number", 1, MAX_CONFLICT_PATCHES)
        source = identifier(data.get("source"), "Chosen capture")
        if source == first_id:
            choices[patch] = 0
        elif source == second_id:
            choices[patch] = 1
        else:
            raise Problem("A conflict choice must name one of the two selected captures.")
    return choices


def _adopted_index(first_codes: np.ndarray, second_codes: np.ndarray,
                   conflict_labels: np.ndarray, label_choices: dict[int, int]) -> np.ndarray:
    """Source index (0/1) adopted for every cell; -1 where neither saw evidence."""
    first_seen = first_codes != MISSING
    second_seen = second_codes != MISSING
    first_decisive = np.isin(first_codes, DECISIVE)
    second_decisive = np.isin(second_codes, DECISIVE)
    adopted = np.full(first_codes.shape, -1, dtype=np.int8)
    # A direction seen by only one camera, even uncertainly, adopts that camera.
    adopted[first_seen] = 0
    adopted[second_seen] = 1
    # Directions both cameras measured default to the first capture; an explicit
    # choice on a disagreement patch moves the whole patch to the second capture.
    both = first_seen & second_seen
    adopted[both] = 0
    for label, source in label_choices.items():
        if source == 1:
            adopted[conflict_labels == label] = 1
    adopted[~(first_seen | second_seen)] = -1
    return adopted


def _merge_codes(first_codes: np.ndarray, second_codes: np.ndarray, adopted: np.ndarray) -> np.ndarray:
    merged = np.full(first_codes.shape, MISSING, dtype=np.uint8)
    use_first = adopted == 0
    use_second = adopted == 1
    merged[use_first] = first_codes[use_first]
    merged[use_second] = second_codes[use_second]
    return merged


def _provenance(first_codes: np.ndarray, second_codes: np.ndarray,
                conflict_labels: np.ndarray, adopted: np.ndarray) -> np.ndarray:
    first_decisive = np.isin(first_codes, DECISIVE)
    second_decisive = np.isin(second_codes, DECISIVE)
    both = first_decisive & second_decisive
    conflict = conflict_labels > 0
    prov = np.full(first_codes.shape, PROV_NOTHING, dtype=np.uint8)
    prov[first_decisive & ~second_decisive] = PROV_FIRST_ONLY
    prov[second_decisive & ~first_decisive] = PROV_SECOND_ONLY
    agree_both = both & ~conflict
    prov[agree_both & (adopted == 0)] = PROV_AGREED_FIRST
    prov[agree_both & (adopted == 1)] = PROV_AGREED_SECOND
    prov[conflict & (adopted == 0)] = PROV_CONFLICT_FIRST
    prov[conflict & (adopted == 1)] = PROV_CONFLICT_SECOND
    return prov


def _patch_order(conflict_labels: np.ndarray) -> list[int]:
    """Connected-patch labels ordered largest first, matching their review rank."""
    populations = np.bincount(conflict_labels.reshape(-1))
    return sorted((i for i in range(1, len(populations)) if populations[i]),
                  key=lambda i: (-populations[i], i))[:MAX_CONFLICT_PATCHES]


def _patch_rows(grid, conflict_labels: np.ndarray, first_codes: np.ndarray, second_codes: np.ndarray,
                adopted: np.ndarray, choices: dict[int, int]) -> list[dict]:
    rows = []
    populations = np.bincount(conflict_labels.reshape(-1))
    solid_angle = grid.solid_angle_per_cell
    for rank, label in enumerate(_patch_order(conflict_labels), 1):
        cells = conflict_labels == label
        directions = grid.directions[cells]
        chosen_index = int(adopted[cells][0])
        rows.append({
            "patch": rank,
            "label": int(label),
            "cells": int(populations[label]),
            "solid_angle_sr": number(populations[label] * solid_angle, 7),
            "center_bearing": number(float(directions[:, 0].mean()) % 360, 4),
            "center_elevation": number(float(directions[:, 1].mean()), 4),
            "first_capture_class": LABELS[int(first_codes[cells][0])],
            "second_capture_class": LABELS[int(second_codes[cells][0])],
            "first_open_cells": int(np.count_nonzero(first_codes[cells] == OPEN)),
            "second_open_cells": int(np.count_nonzero(second_codes[cells] == OPEN)),
            "chosen_source": "first" if chosen_index == 0 else "second",
            "reviewed": rank in choices,
        })
    return rows


def _merge_rgb(earlier_rgb: np.ndarray, later_rgb: np.ndarray, adopted: np.ndarray) -> np.ndarray:
    rgb = np.full(earlier_rgb.shape, np.nan, dtype=np.float32)
    rgb[adopted == 0] = earlier_rgb[adopted == 0]
    rgb[adopted == 1] = later_rgb[adopted == 1]
    return rgb


def _conflict_review_map(provenance: np.ndarray, conflict_labels: np.ndarray,
                         rank_of_label: dict[int, int], grid) -> Image.Image:
    """Provenance colors with every disagreement patch outlined for review."""
    chart = angular_map(PROV_PALETTE[provenance], grid,
                        "Disagreement patches awaiting or holding a choice", PROV_LEGEND)
    draw = ImageDraw.Draw(chart)
    x0, y0, width, height = 69, 57, 835, 352
    rows, cols = conflict_labels.shape
    populations = np.bincount(conflict_labels.reshape(-1))
    for label, rank in rank_of_label.items():
        if not populations[label]:
            continue
        rr, cc = np.nonzero(conflict_labels == label)
        # angular_map flips the image vertically before resizing.
        px = x0 + cc.astype(float) / cols * width
        py = y0 + (rows - 1 - rr).astype(float) / rows * height
        left, right = px.min() - 2, px.max() + 2
        top, bottom = py.min() - 2, py.max() + 2
        draw.rectangle((left, top, right, bottom), outline=GOLD, width=2)
        draw.text((left + 2, top - 12), str(rank), fill=PALE, font=_draw_font(13))
    return chart


PROV_NAMES = {
    PROV_NOTHING: "neither_capture",
    PROV_FIRST_ONLY: "first_only",
    PROV_SECOND_ONLY: "second_only",
    PROV_AGREED_FIRST: "agreed_first_adopted",
    PROV_AGREED_SECOND: "agreed_second_adopted",
    PROV_CONFLICT_FIRST: "conflict_first_chosen",
    PROV_CONFLICT_SECOND: "conflict_second_chosen",
}


def _provenance_rows(grid, adopted: np.ndarray, provenance: np.ndarray, merged: np.ndarray,
                     first_id: str, second_id: str) -> list[dict]:
    rows = []
    directions = grid.directions
    for index, ((bearing, elevation), source_index, prov, code) in enumerate(zip(
            directions.reshape(-1, 2), adopted.reshape(-1), provenance.reshape(-1), merged.reshape(-1), strict=True)):
        rows.append({
            "cell": index + 1,
            "bearing": number(float(bearing), 5),
            "elevation": number(float(elevation), 5),
            "adopted_capture_id": first_id if source_index == 0 else second_id if source_index == 1 else None,
            "provenance": PROV_NAMES[int(prov)],
            "merged_class": LABELS[int(code)],
        })
    return rows


def _area_reading(codes: np.ndarray, solid_angle: float) -> dict:
    counts = np.bincount(codes.reshape(-1), minlength=len(LABELS))
    total = codes.size
    observed = total - counts[MISSING]
    decisive = counts[OPEN] + counts[OCCLUDED]
    return {
        "cells_total": total,
        "cells_with_camera_evidence": int(observed),
        "cells_decisive": int(decisive),
        "solid_angle_sr": number(total * solid_angle, 7),
        "percent_covered_by_either_capture": number(100 * observed / total),
        "percent_open_of_decisive": number(100 * counts[OPEN] / decisive) if decisive else None,
        "open_sky_bounds_percent_of_footprint": [
            number(100 * counts[OPEN] / total),
            number(100 * (counts[OPEN] + counts[MISSING] + counts[UNCERTAIN]) / total),
        ],
    }


def compute_combined(grid, first, second, rank_choices: dict[int, int]) -> dict:
    """Deterministic merge core, reusable for retained-record point probes."""
    first_sample = first.sample(grid)
    second_sample = second.sample(grid)
    first_codes, second_codes = first_sample.codes, second_sample.codes
    disagreement = (np.isin(first_codes, DECISIVE) & np.isin(second_codes, DECISIVE)
                    & (first_codes != second_codes))
    conflict_labels = _connected_conflicts(disagreement, grid.periodic)
    ordered_labels = _patch_order(conflict_labels)
    rank_of_label = {label: rank for rank, label in enumerate(ordered_labels, 1)}
    label_of_rank = {rank: label for label, rank in rank_of_label.items()}
    unknown = sorted(set(rank_choices) - set(label_of_rank))
    if unknown:
        raise Problem(f"Conflict patch {unknown[0]} does not exist at this sampling density. Reopen the preview.")
    label_choices = {label_of_rank[rank]: source for rank, source in rank_choices.items()}
    adopted = _adopted_index(first_codes, second_codes, conflict_labels, label_choices)
    merged = _merge_codes(first_codes, second_codes, adopted)
    provenance = _provenance(first_codes, second_codes, conflict_labels, adopted)
    patch_rows = _patch_rows(grid, conflict_labels, first_codes, second_codes, adopted, rank_choices)
    return {
        "grid": grid, "first_sample": first_sample, "second_sample": second_sample,
        "first_codes": first_codes, "second_codes": second_codes,
        "disagreement": disagreement, "conflict_labels": conflict_labels,
        "rank_of_label": rank_of_label, "adopted": adopted, "merged": merged,
        "provenance": provenance, "patch_rows": patch_rows,
    }


def combined_view(bench: Bench) -> Result:
    first, second = bench.two()
    data = bench.order.settings
    fields(data, "conflict_choices")
    rank_choices = _parse_choices(data.get("conflict_choices", []), first.id, second.id)
    grid = bench.grid
    core = compute_combined(grid, first, second, rank_choices)
    first_sample, second_sample = core["first_sample"], core["second_sample"]
    first_codes, second_codes = core["first_codes"], core["second_codes"]
    disagreement, conflict_labels = core["disagreement"], core["conflict_labels"]
    rank_of_label, adopted, merged, provenance, patch_rows = (
        core["rank_of_label"], core["adopted"], core["merged"], core["provenance"], core["patch_rows"])

    report = Result(
        "combined-view", "Combined canopy view from two captures",
        "Each direction is counted once. Directions seen by only one camera adopt that camera; "
        "matching readings merge silently; disagreement patches wait for an explicit capture choice. "
        "Cells neither camera reached stay empty.",
    )
    coverage = _area_reading(merged, grid.solid_angle_per_cell)
    both_decisive = int(np.count_nonzero(np.isin(first_codes, DECISIVE) & np.isin(second_codes, DECISIVE)))
    report.metrics = {
        **coverage,
        "conflict_patches": len(patch_rows),
        "conflict_cells": int(np.count_nonzero(disagreement)),
        "decisive_cells_in_both_captures": both_decisive,
        "first_capture_id": first.id,
        "second_capture_id": second.id,
        "first_capture_title": first.title,
        "second_capture_title": second.title,
        "sampling": grid.describe(),
    }
    report.readings = patch_rows
    report.png("combined-sky-map", "Merged sky class map", category_map(merged, grid, "Combined sky by solid angle"))
    report.png("source-provenance", "Which capture each direction came from",
               angular_map(PROV_PALETTE[provenance], grid, "Source provenance of the combined view", PROV_LEGEND))
    if patch_rows:
        report.png("conflict-review", "Disagreement patches to choose between",
                   _conflict_review_map(provenance, conflict_labels, rank_of_label, grid))
    composite_rgb = _merge_rgb(first_sample.rgb, second_sample.rgb, adopted)
    display = np.where(np.isnan(composite_rgb).any(axis=-1, keepdims=True),
                       np.asarray(PROV_PALETTE[PROV_NOTHING], np.float32) / 255,
                       np.clip(composite_rgb, 0, 1))
    report.png("combined-photo", "Composite photograph in angular coordinates",
               angular_map(np.rint(display * 255).astype(np.uint8), grid,
                           "Adopted camera evidence in bearing and elevation"))
    report.csv("conflict-patches", "Disagreement patch decisions", patch_rows)
    report.csv("cell-provenance", "Adopted capture for every sampled direction",
               _provenance_rows(grid, adopted, provenance, merged, first.id, second.id))
    report.warnings = [
        *first.method_warnings(), *second.method_warnings(),
        "The merge assumes both calibrated lenses refer to the same station directions; verify lens residuals before trusting overlaps.",
    ]
    if patch_rows and not all(row["reviewed"] for row in patch_rows):
        report.warnings.append(
            f"{sum(not row['reviewed'] for row in patch_rows)} disagreement patch(es) still use the first capture by default. "
            "Choose a capture for each patch before retaining the combined sky share.")
    report.probe_kind = "angular-cell"
    return report


def adopted_source_for_direction(grid, first, second, conflict_choices: list[dict],
                                 direction: tuple[float, float]) -> dict:
    """Recompute the merge and report which capture supplies one direction."""
    rank_choices = _parse_choices(conflict_choices or [], first.id, second.id)
    core = compute_combined(grid, first, second, rank_choices)
    cell = grid.index(direction)
    if cell is None:
        raise Problem("That bearing and elevation fall outside the retained direction window.")
    row, col = cell
    source_index = int(core["adopted"][row, col])
    if source_index < 0:
        raise Problem("Neither retained camera measured that direction.")
    return {
        "source_index": source_index,
        "provenance": PROV_NAMES[int(core["provenance"][row, col])],
        "merged_class": LABELS[int(core["merged"][row, col])],
        "grid_cell": [row, col],
    }
