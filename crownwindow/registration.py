"""Register two repeat photographs with unchanged wall corners before diffing.

The pair is compared on a common native-pixel frame. Each marked corner pair
proposes one small in-plane shift; when the proposals do not share a single
movement the report keeps the untouched comparison and a nearby-only shift as
explicit alternatives, and sends every pixel too far from a trusted corner to
a separate review list. No corner is ever assumed to hold outside its trust
radius, and every change reading names the corner that supported it.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .bench import Bench, Source
from .camera import encode_png, rgb_luminance, srgb_to_linear
from .contracts import Problem, fields, items, mapping, number, point, real, text
from .field import image_readout, native_witness
from .result import Result
from .visual import (DARK, GOLD, MINT, MUTED, PALE, PINK, _draw_font, comparison_panel,
                     height_chart, photo_canvas)


FRAME_EDGE = 920
REVIEW_COLOR = (109, 104, 125)
BASE_COLOR = (38, 44, 55)
NEGATIVE = (250, 125, 143)
POSITIVE = (122, 214, 163)
STRAINED_COLOR = (211, 159, 80)


@dataclass(frozen=True)
class Corner:
    label: str
    first: tuple[float, float]
    second: tuple[float, float]
    band: str
    support: float

    @classmethod
    def parse(cls, raw: dict, index: int) -> Corner:
        data = mapping(raw, "Wall corner")
        fields(data, "label", "first", "second", "band", "support")
        band = data.get("band", "near")
        if band not in ("near", "far"):
            raise Problem("Each corner band must be near or far.")
        return cls(
            text(data.get("label") or f"Corner {index + 1}", "Corner label", 65),
            point(data.get("first"), "First photograph corner"),
            point(data.get("second"), "Second photograph corner"),
            band,
            real(data.get("support", 1), "Confidence in the corner", .05, 1),
        )


DEMO_CORNERS = [
    {"label": "Near wall corner", "first": [.42, .62], "second": [.428, .612],
     "band": "near", "support": .95},
    {"label": "Far stone corner", "first": [.66, .34], "second": [.666, .335],
     "band": "far", "support": .8},
    {"label": "Wall base corner", "first": [.31, .74], "second": [.318, .733],
     "band": "near", "support": .9},
]


def _frame_shape(first: Source) -> tuple[int, int]:
    width = min(FRAME_EDGE, first.camera.width)
    height = max(24, round(width * first.camera.height / first.camera.width))
    return width, height


def _frame_points(corners: list[Corner], source: Source, shape: tuple[int, int], key: str) -> np.ndarray:
    native = np.array([getattr(corner, key) for corner in corners], dtype=float)
    pixels = native * np.array([source.camera.width - 1, source.camera.height - 1], dtype=float)
    scale = np.array([shape[0] - 1, shape[1] - 1], dtype=float) / np.array(
        [source.camera.width - 1, source.camera.height - 1], dtype=float)
    return pixels * scale


def _sample_frame(source: Source, shape: tuple[int, int], shift_xy: tuple[float, float]):
    """Resample the source onto the common frame; the shift moves first onto second."""
    width, height = shape
    gx, gy = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    coords = np.stack((gx - shift_xy[0], gy - shift_xy[1]), axis=-1)
    rgb, seen = source.camera.bilinear_pixels(coords)
    return np.clip(rgb, 0, 1), seen


def _fit_shift(first_pts: np.ndarray, second_pts: np.ndarray,
               weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean shift in frame pixels, plus each point's residual."""
    deltas = second_pts - first_pts
    shift = np.average(deltas, axis=0, weights=weights)
    residuals = np.linalg.norm(deltas - shift, axis=1)
    return shift, residuals


def _basis(name: str, label: str, shift, corners: list[Corner], residuals,
           marking_px: float, note: str) -> dict:
    if shift is None:
        return {"id": name, "label": label, "shift_px": None, "corner_count": 0,
                "max_residual_px": None, "status": "untouched", "note": note, "corner_labels": []}
    status = "one_move" if float(np.max(residuals)) <= marking_px else "strained"
    return {
        "id": name, "label": label, "shift_px": [number(v, 4) for v in shift],
        "max_residual_px": number(float(np.max(residuals)), 4),
        "corner_count": len(corners), "status": status, "note": note,
        "corner_labels": [corner.label for corner in corners],
    }


def _support_map(shape: tuple[int, int], points: np.ndarray, radius_px: float) -> np.ndarray:
    if not len(points):
        return np.zeros((shape[1], shape[0]), dtype=bool)
    gx, gy = np.meshgrid(np.arange(shape[0], dtype=float), np.arange(shape[1], dtype=float))
    cover = np.zeros((shape[1], shape[0]), dtype=bool)
    for px, py in points:
        cover |= (gx - px) ** 2 + (gy - py) ** 2 <= radius_px ** 2
    return cover


def _difference_rgb(rgb_a: np.ndarray, rgb_b: np.ndarray, joint: np.ndarray,
                    review: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    luma_a = rgb_luminance(srgb_to_linear(rgb_a))
    luma_b = rgb_luminance(srgb_to_linear(rgb_b))
    signed = luma_b - luma_a
    magnitude = np.clip(np.abs(signed) / max(floor * 5, .01), 0, 1)
    tone = np.where((signed >= 0)[..., None], POSITIVE, NEGATIVE)
    rendered = np.empty(rgb_a.shape, dtype=np.uint8)
    rendered[:] = np.array(BASE_COLOR, dtype=float) + (tone - np.array(BASE_COLOR, dtype=float)) * magnitude[..., None]
    rendered[~joint] = BASE_COLOR
    # Only jointly measured, actually differing cells beyond corner support are held for review.
    review_cells = review & joint & (np.abs(signed) >= floor)
    rendered[review_cells] = REVIEW_COLOR
    return signed, rendered


def _corner_overlay(source: Source, corners: list[Corner], key: str, shape: tuple[int, int],
                    residuals: np.ndarray, marking_px: float) -> Image.Image:
    points = _frame_points(corners, source, shape, key)
    image = photo_canvas(source.camera, FRAME_EDGE).resize(shape, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    for index, ((x, y), corner) in enumerate(zip(points, corners, strict=True), 1):
        tone = STRAINED_COLOR if residuals[index - 1] > marking_px else GOLD
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=tone, outline=DARK, width=2)
        draw.text((x + 10, y - 11), f"{index} {corner.label[:18]}", fill=PALE,
                  stroke_width=2, stroke_fill=DARK, font=_draw_font(14))
    return image


def _change_components(signed: np.ndarray, changed: np.ndarray, trusted: np.ndarray,
                       joint: np.ndarray, basis_id: str, anchor_points: np.ndarray,
                       anchor_labels: list[str], shape: tuple[int, int]) -> list[dict]:
    mask = changed & trusted & joint
    labels, count = ndimage.label(mask)
    rows = []
    for component in range(1, count + 1):
        members = np.argwhere(labels == component)
        if len(members) < 5:
            continue
        iy, ix = members[:, 0], members[:, 1]
        centroid = members.mean(axis=0)
        rep = members[np.argmin(((members - centroid) ** 2).sum(axis=1))]
        if len(anchor_points):
            gap = np.linalg.norm(anchor_points - np.array([rep[1], rep[0]]), axis=1)
            anchor_index = int(np.argmin(gap))
            anchor_label = anchor_labels[anchor_index]
            anchor_gap = float(gap[anchor_index])
        else:
            anchor_index, anchor_label, anchor_gap = -1, None, None
        local = signed[iy, ix]
        mean_delta = float(local.mean())
        rows.append({
            "basis": basis_id,
            "state": "later_darker" if mean_delta < 0 else "later_brighter",
            "frame_cells": int(len(members)),
            "frame_position": [number(rep[1] / (shape[0] - 1), 6),
                               number(rep[0] / (shape[1] - 1), 6)],
            "mean_luminance_delta": number(mean_delta, 6),
            "extreme_delta": number(float(local[np.argmax(np.abs(local))]), 6),
            "reference_corner": anchor_label,
            "reference_corner_index": anchor_index + 1 if anchor_index >= 0 else None,
            "distance_to_reference_px": number(anchor_gap, 3),
        })
    rows.sort(key=lambda row: -row["frame_cells"])
    return rows


def _annotate_map(image: Image.Image, rows: list[dict], shape: tuple[int, int], title: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 46), DARK)
    canvas.paste(image, (0, 46))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 12), title, fill=PALE, font=_draw_font(20))
    for row in rows[:8]:
        x = row["frame_position"][0] * (shape[0] - 1)
        y = 46 + row["frame_position"][1] * (shape[1] - 1)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=PALE, width=2)
    return canvas


def _residual_chart(corners: list[Corner], global_residuals: np.ndarray, near_residuals,
                    near_indices: list[int], marking_px: float) -> Image.Image:
    near_lookup = {global_index: local for local, global_index in enumerate(near_indices)}
    near_values = [number(near_residuals[near_lookup[i]], 4) if i in near_lookup else None
                   for i in range(len(corners))]
    chart = height_chart([
        {"name": "Global move residual", "color": PINK,
         "values": [number(v, 4) for v in global_residuals]},
        {"name": "Near-only residual (x = not a near corner)", "color": MINT, "values": near_values},
    ], "Residual after fitting (frame pixels)", "Pixels", "Marked wall corners",
        reference=marking_px)
    draw = ImageDraw.Draw(chart)
    count = max(1, len(corners) - 1)
    plot_x = 72 + np.arange(len(corners)) * (788 / count)
    for i, x in enumerate(plot_x):
        if i in near_lookup:
            continue
        y = 67 + 260 * .15
        draw.line((x - 5, y - 5, x + 5, y + 5), fill=MUTED, width=2)
        draw.line((x - 5, y + 5, x + 5, y - 5), fill=MUTED, width=2)
    return chart


def wall_registration(bench: Bench) -> Result:
    first, second = bench.two()
    data = bench.order.settings
    fields(data, "corners", "difference_floor", "marking_radius_px", "trust_radius_fraction")
    corners = [Corner.parse(raw, i) for i, raw in enumerate(
        items(data.get("corners", DEMO_CORNERS), "Wall corners", 2, 12))]
    floor = real(data.get("difference_floor", .08), "Luminance change indicator", .005, .60)
    marking_px = real(data.get("marking_radius_px", 6), "Corner marking radius in frame pixels", 1, 40)
    trust_fraction = real(data.get("trust_radius_fraction", .30),
                          "Trust radius as a fraction of frame width", .08, .95)
    shape = _frame_shape(first)
    first_pts = _frame_points(corners, first, shape, "first")
    second_pts = _frame_points(corners, second, shape, "second")
    weights = np.array([corner.support for corner in corners])

    global_shift, global_residuals = _fit_shift(first_pts, second_pts, weights)
    near_indices = [i for i, corner in enumerate(corners) if corner.band == "near"]
    near_shift = near_residuals = None
    near_corners: list[Corner] = []
    if near_indices:
        near_shift, near_residuals = _fit_shift(
            first_pts[near_indices], second_pts[near_indices], weights[near_indices])
        near_corners = [corners[i] for i in near_indices]

    bases = [
        _basis("untouched", "Untouched original comparison", None, [], None, marking_px,
               "Both photographs compared on their own pixels without any corner adjustment."),
        _basis("global", "One move from every wall corner", global_shift, corners,
               global_residuals, marking_px,
               "Single in-plane shift from all marked corners; strained residuals mean one move is not supported."),
    ]
    if near_shift is not None:
        bases.append(_basis("near_only", "Trust near-wall corners only", near_shift, near_corners,
                            near_residuals, marking_px,
                            "Shift from near-band corners only; far parts of the scene stay outside its support."))

    trust_radius = trust_fraction * shape[0]
    global_cover = _support_map(shape, first_pts, trust_radius)
    near_cover = (_support_map(shape, first_pts[near_indices], trust_radius)
                  if near_indices else np.zeros((shape[1], shape[0]), bool))

    first_raw, first_seen = _sample_frame(first, shape, (0, 0))
    second_raw, second_seen = _sample_frame(second, shape, (0, 0))
    joint_raw = first_seen & second_seen
    signed_raw, map_raw = _difference_rgb(first_raw, second_raw, joint_raw, ~global_cover, floor)
    changed_raw = joint_raw & (np.abs(signed_raw) >= floor)

    shifted_first, shifted_seen = _sample_frame(first, shape, tuple(global_shift))
    joint_global = shifted_seen & second_seen
    signed_global, map_global = _difference_rgb(shifted_first, second_raw, joint_global,
                                                 ~global_cover, floor)
    changed_global = joint_global & (np.abs(signed_global) >= floor)

    if near_shift is not None:
        near_first, near_seen = _sample_frame(first, shape, tuple(near_shift))
        joint_near = near_seen & second_seen
        signed_near, map_near = _difference_rgb(near_first, second_raw, joint_near, ~near_cover, floor)
        changed_near = joint_near & (np.abs(signed_near) >= floor)
    else:
        signed_near = map_near = changed_near = joint_near = None

    anchor_labels = [corner.label for corner in corners]
    rows_raw = _change_components(signed_raw, changed_raw, global_cover, joint_raw,
                                  "untouched", first_pts, anchor_labels, shape)
    rows_global = _change_components(signed_global, changed_global, global_cover, joint_global,
                                     "global", first_pts, anchor_labels, shape)
    rows_near = (_change_components(signed_near, changed_near, near_cover, joint_near,
                                    "near_only", first_pts[near_indices],
                                    [anchor_labels[i] for i in near_indices], shape)
                 if near_shift is not None else [])

    pending = int(np.count_nonzero(~global_cover & joint_global & (np.abs(signed_global) >= floor)))
    trusted_changes = int(np.count_nonzero(changed_global & global_cover))
    recommendation = "global" if bases[1]["status"] == "one_move" else (
        "near_only" if near_shift is not None and bases[2]["status"] == "one_move" else "untouched")
    near_lookup = {global_index: local for local, global_index in enumerate(near_indices)}

    report = Result("repeat-align", "Repeat change after wall-corner alignment",
                    "Keep the untouched comparison beside corner-adjusted views, and only trust a change where a marked corner supports it.")
    report.metrics = {
        "frame_px": list(shape),
        "corner_count": len(corners),
        "near_corner_count": len(near_indices),
        "global_shift_px": bases[1]["shift_px"],
        "global_max_residual_px": bases[1]["max_residual_px"],
        "global_status": bases[1]["status"],
        "marking_radius_px": marking_px,
        "trust_radius_px": number(trust_radius, 3),
        "difference_floor": floor,
        "trusted_change_cells": trusted_changes,
        "pending_review_cells": pending,
        "recommended_basis": recommendation,
        "bases": bases,
    }
    corner_rows = [
        {"kind": "corner", "index": i + 1, "label": corner.label, "band": corner.band,
         "first_unit": list(corner.first), "second_unit": list(corner.second),
         "global_residual_px": number(float(global_residuals[i]), 4),
         "near_residual_px": (number(float(near_residuals[near_lookup[i]]), 4)
                              if i in near_lookup else None),
         "used_by": [basis["id"] for basis in bases if corner.label in basis["corner_labels"]]}
        for i, corner in enumerate(corners)
    ]
    change_rows = [dict(row, kind="change")
                   for row in rows_raw[:24] + rows_global[:40] + rows_near[:40]]
    report.readings = corner_rows + change_rows

    report.png("untouched-change", "Untouched comparison: later minus earlier",
               _annotate_map(Image.fromarray(map_raw), rows_raw, shape,
                             "Untouched original comparison (mauve = outside corner support)"))
    report.png("aligned-change", "Corner-adjusted change on the global move",
               _annotate_map(Image.fromarray(map_global), rows_global, shape,
                             "All-corner shift; mauve cells are listed for review, not called changes"))
    if map_near is not None:
        report.png("near-change", "Change supported by near-wall corners only",
                   _annotate_map(Image.fromarray(map_near), rows_near, shape,
                                 "Near-corner shift; everything outside the near-corner radius is review-only"))
    report.png("corner-marks", "Numbered wall corners on both originals",
               comparison_panel([
                   (first.title, _corner_overlay(first, corners, "first", shape, global_residuals, marking_px)),
                   (second.title, _corner_overlay(second, corners, "second", shape, global_residuals, marking_px)),
               ], "Matching wall corners (gold = within marking radius, amber = strained)"))
    report.png("corner-residuals", "Corner residuals against each fitted move",
               _residual_chart(corners, global_residuals, near_residuals, near_indices, marking_px))
    report.csv("corner-fit", "Corner shifts and residuals", corner_rows)
    report.csv("aligned-changes", "Change components and their reference corners", change_rows)
    report.warnings = [
        "Alignment fits one in-plane shift only. A tilted camera or a moved station needs fresh lens placement, not this tool.",
        "Mauve cells are farther than the trust radius from every marked corner and are held out of confirmed changes.",
        "Camera color is uncalibrated; weather and exposure can produce luminance differences without a physical change.",
    ]
    report.probe_kind = "align"
    return report


def alignment_probe(first: Source, second: Source, settings: dict, frame_point,
                    basis_id: str, radius: float) -> dict:
    """Open the actual pixels behind one change-map point on both originals."""
    data = mapping(settings, "Alignment settings")
    fields(data, "corners", "difference_floor", "marking_radius_px", "trust_radius_fraction")
    corners = [Corner.parse(raw, i) for i, raw in enumerate(
        items(data.get("corners", DEMO_CORNERS), "Wall corners", 2, 12))]
    trust_fraction = real(data.get("trust_radius_fraction", .30),
                          "Trust radius as a fraction of frame width", .08, .95)
    shape = _frame_shape(first)
    first_pts = _frame_points(corners, first, shape, "first")
    second_pts = _frame_points(corners, second, shape, "second")
    weights = np.array([corner.support for corner in corners])
    global_shift, global_residuals = _fit_shift(first_pts, second_pts, weights)
    near_indices = [i for i, corner in enumerate(corners) if corner.band == "near"]
    near_shift = None
    if near_indices:
        near_shift, _ = _fit_shift(first_pts[near_indices], second_pts[near_indices],
                                   weights[near_indices])
    shifts = {"untouched": None, "global": global_shift,
              "near_only": near_shift if near_shift is not None else global_shift}
    if basis_id not in shifts:
        raise Problem("Choose the untouched, global or near-only basis for this point.")
    shift_xy = shifts[basis_id]

    uv = point(frame_point, "Change-map point")
    fx, fy = uv
    if shift_xy is None:
        first_point = [fx, fy]
    else:
        first_point = [max(0.0, min(1.0, fx - float(shift_xy[0]) / (shape[0] - 1))),
                       max(0.0, min(1.0, fy - float(shift_xy[1]) / (shape[1] - 1)))]
    first_rows = image_readout(first, [first_point], radius)
    second_rows = image_readout(second, [list(uv)], radius)

    allowed = near_indices if basis_id == "near_only" else list(range(len(corners)))
    gap = np.linalg.norm(first_pts[allowed] - np.array([fx * (shape[0] - 1), fy * (shape[1] - 1)]),
                        axis=1)
    corner_index = allowed[int(np.argmin(gap))]
    trust_radius = trust_fraction * shape[0]
    corner = corners[corner_index]

    def witness_asset(source: Source, readout: dict, name: str, label: str) -> dict:
        return {"name": name, "mime": "image/png", "label": label,
                "data": base64.b64encode(encode_png(native_witness(source, readout, radius))).decode("ascii")}

    return {
        "basis": basis_id,
        "frame_point": list(uv),
        "first": {"source": first.public(), "readout": first_rows[0],
                  "witness": witness_asset(first, first_rows[0], "first-evidence.png",
                                           "Actual earlier original at this point")},
        "second": {"source": second.public(), "readout": second_rows[0],
                   "witness": witness_asset(second, second_rows[0], "second-evidence.png",
                                            "Actual later original at this point")},
        "reference_corner": {"index": corner_index + 1, "label": corner.label, "band": corner.band,
                             "first_unit": list(corner.first), "second_unit": list(corner.second),
                             "distance_px": number(float(gap.min()), 3),
                             "trust_radius_px": number(trust_radius, 3),
                             "within_trust": bool(float(gap.min()) <= trust_radius),
                             "global_residual_px": number(float(global_residuals[corner_index]), 4)},
        "shift_px": None if shift_xy is None else [number(v, 4) for v in shift_xy],
    }
