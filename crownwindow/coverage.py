"""Image support, spherical area readings and horizon profiles."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .bench import Bench
from .calibration import lens_limits
from .camera import bounded_size, describe_rgb, display_scale, rgb_luminance, srgb_to_linear
from .contracts import Footprint, Line, Problem, boolean, fields, number, real, whole
from .projection import Grid
from .result import Result
from .sky import LABELS, MISSING, OCCLUDED, OPEN, UNCERTAIN, native_classes, state_counts
from .visual import (GOLD, MINT, PINK, angular_map, calibration_view, category_map, comparison_panel,
                     height_chart, normalized_heat, photo_canvas, tinted_rgb)


def inspect_source(bench: Bench) -> Result:
    source = bench.one()
    camera = source.camera
    data = bench.order.settings
    fields(data, "floor", "ceiling", "gamma", "monochrome", "threshold")
    floor = real(data.get("floor", 0), "Display floor", 0, .95)
    ceiling = real(data.get("ceiling", 1), "Display ceiling", .05, 1)
    gamma = real(data.get("gamma", 1), "Display gamma", .25, 4)
    monochrome = boolean(data.get("monochrome", False), "Monochrome display")
    guard = real(data.get("threshold", .985), "Clipping level", .65, 1)
    if ceiling - floor < .025:
        raise Problem("Separate the two display bounds by at least .025.")
    rgb = display_scale(camera, floor, ceiling, gamma, monochrome=monochrome)
    working = Image.fromarray(rgb).resize(bounded_size(camera.width, camera.height, 880), Image.Resampling.LANCZOS)
    result = Result("source-check", "Source signal and clipping",
                    "Display settings affect the picture only. The inspection numbers are read from the camera image.")
    result.metrics = {"source_px": [camera.width, camera.height],
                      "opaque_percent": number(100 * (1 - camera.invalid_share)),
                      "clipping_threshold": guard, "channels": describe_rgb(camera.color[camera.valid][::2])}
    ranges = _intake_rows(camera.color, camera.valid, guard)
    result.readings = ranges
    result.metrics["clipped_fraction"] = {row["channel"]: row["fraction_clipped"] for row in ranges}
    result.png("camera-display", "Display window", working)
    problem = _clip_overlay(camera, guard)
    result.png("camera-clipping", "Clipping versus missing alpha", problem)
    hist = _intake_histogram(camera.color, camera.valid, guard)
    result.png("source-distribution", "RGB distribution", hist)
    result.csv("camera-signal", "Camera signal table", ranges)
    result.probe_kind = "image"
    return result


def _intake_rows(colors: np.ndarray, valid: np.ndarray, limit: float) -> list[dict]:
    rows = []
    for index, name in enumerate(("red", "green", "blue")):
        values = colors[..., index][valid]
        clipped = np.count_nonzero(values >= limit)
        rows.append({
            "channel": name, "measured": len(values), "clipped": int(clipped),
            "fraction_clipped": number(clipped / len(values)),
            "mean": number(values.mean()), "std": number(values.std()),
            "p05": number(np.quantile(values, .05)), "median": number(np.median(values)),
            "p95": number(np.quantile(values, .95)),
        })
    return rows


def _clip_overlay(camera, guard: float) -> Image.Image:
    base = photo_canvas(camera, 880)
    support = Image.fromarray(camera.valid.astype(np.uint8) * 255).resize(base.size, Image.Resampling.NEAREST)
    masks = []
    for i in range(3):
        clipped = camera.valid & (camera.color[..., i] >= guard)
        masks.append(np.asarray(Image.fromarray(clipped.astype(np.uint8) * 255).resize(base.size, Image.Resampling.NEAREST)) > 0)
    plane = np.array(base) // 2
    directions = np.array([[237, 93, 76], [125, 217, 147], [110, 172, 247]], dtype=np.int16)
    flags = np.stack(masks, axis=-1)
    tinted = (flags.astype(np.int16)[..., None] * directions).sum(axis=-2)
    seen = np.asarray(support) > 0
    marked = np.where(flags.any(axis=-1, keepdims=True), np.minimum(255, tinted), plane)
    marked[~seen] = (24, 38, 49)
    return Image.fromarray(marked.astype(np.uint8))


def _intake_histogram(colors: np.ndarray, valid: np.ndarray, guard: float) -> Image.Image:
    entries = []
    for i, (label, tone) in enumerate((("Red", PINK), ("Green", MINT), ("Blue", (132, 186, 242)))):
        values = colors[..., i][valid][::2]
        counts, _ = np.histogram(values, bins=120, range=(0, 1))
        entries.append({"name": label, "color": tone, "values": np.log10(1 + counts).tolist()})
    return height_chart(entries, "Camera code distribution", "log counts", f"0–1 display code • clipping at {guard:g}")


def line_profile(bench: Bench) -> Result:
    source = bench.one()
    data = bench.order.settings
    fields(data, "line", "samples", "width")
    line = Line.parse(data.get("line", [[.16, .42], [.84, .42]]))
    count = whole(data.get("samples", 121), "Positions", 12, 320)
    width = real(data.get("width", 0), "Mean band width in native pixels", 0, 85)
    reports = source.camera.line_readout(line, count, width)
    source_width = source.camera.width - 1
    source_height = source.camera.height - 1
    length = np.hypot((line.end[0] - line.start[0]) * source_width, (line.end[1] - line.start[1]) * source_height)
    support = sum(bool(row["sample_count"]) for row in reports)
    result = Result("line-profile", "Luminance along a source line",
                    "The line reads actual image codes without angular resampling. The plotted range stops at missing support.")
    result.metrics = {"native_length_px": number(length), "positions": count, "positions_with_sample": support,
                      "requested_width_px": width}
    result.readings = reports
    result.png("line-chart", "Observed values by distance", height_chart(
        [{"name": "Luminance", "color": MINT, "values": [r["luminance"] for r in reports]}],
        "Source line", "Linear display luminance", f"{length:.1f} pixels"))
    frame = photo_canvas(source.camera)
    draw = ImageDraw.Draw(frame)
    ranges = [(u * (frame.width - 1), v * (frame.height - 1)) for u, v in line.public()]
    draw.line(ranges, fill=GOLD, width=3)
    for p, label in zip(ranges, ("A", "B"), strict=True):
        x, y = p
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=GOLD)
        draw.text((x + 9, y - 9), label, fill=GOLD)
    result.png("line-on-source", "Where the line was measured", frame)
    result.csv("line-samples", "Native sample readout", reports)
    result.probe_kind = "image"
    return result


def inspect_geometry(bench: Bench) -> Result:
    source = bench.one()
    fields(bench.order.settings)
    trial = lens_limits(source.lens, source.camera)
    _, inside = source.lens.world_to_image(bench.grid.directions, source.camera)
    points, _ = source.lens.world_to_image(bench.grid.directions, source.camera)
    _, valid = source.camera.bilinear(points)
    layer = np.full(inside.shape + (3,), (38, 48, 56), np.uint8)
    layer[inside] = (144, 111, 67)
    layer[inside & valid] = (140, 203, 173)
    report = Result("lens-support", "What the lens actually saw",
                    "The registered lens boundary and the footprint of opaque camera data are separate.")
    rows = trial.pop("rows")
    report.metrics = trial
    report.readings = rows
    report.png("lens-on-photo", "Calibrated guide on camera pixels", calibration_view(source.camera, source.lens))
    report.png("lens-support", "Lens field and opaque coverage", angular_map(layer, bench.grid, "Lens support map",
               {"Camera support": (140, 203, 173), "Lens without pixels": (144, 111, 67), "Outside lens": (38, 48, 56)}))
    report.csv("bearing-coverage", "Usable elevation by bearing", rows)
    report.probe_kind = "image"
    return report


def classify_photo(bench: Bench) -> Result:
    source = bench.one()
    fields(bench.order.settings)
    rgb, label, margin = native_classes(source.camera, source.lens, source.classifier)
    state = state_counts(label)
    result = Result("sky-labels", "Image class and confidence",
                    "Each visible pixel is read as open, obstructed, or uncertain. Missing support stays separate.")
    result.metrics = {"preview": [int(label.shape[1]), int(label.shape[0])],
                      "observations": state, "classifier": source.classifier.describe()}
    result.readings = [{"class": key, **value} for key, value in state.items()]
    result.png("class-overlay", "Classification on the original photograph", tinted_rgb(rgb, label))
    result.png("class-margin", "Positive means closer to sky", normalized_heat(margin, label != MISSING, -1, 1))
    result.csv("class-pixel-summary", "Image class pixel counts", result.readings)
    result.warnings = source.method_warnings()
    result.probe_kind = "image"
    return result


def _area_summary(code: np.ndarray, sr: float) -> dict:
    pieces = state_counts(code, sr)
    n = code.size
    counts = np.bincount(code.reshape(-1), minlength=4)
    observed = n - counts[MISSING]
    known = counts[OPEN] + counts[OCCLUDED]
    return {
        "solid_angle_sr": number(n * sr, 7),
        "percent_camera_support": number(100 * observed / n),
        "percent_decisive": number(100 * known / n),
        "percent_open_of_decisive": number(100 * counts[OPEN] / known) if known else None,
        "bounds_open_percent_of_full_footprint": [number(100 * counts[OPEN] / n),
                number(100 * (counts[OPEN] + counts[MISSING] + counts[UNCERTAIN]) / n)],
        "classes": pieces,
    }


def _sector_rows(bench: Bench, labels: np.ndarray, across: int, up: int) -> list[dict]:
    grid = bench.grid
    bearings = grid.bearing_edges
    altitudes = grid.cell_elevations
    rows = []
    for i, vertical in enumerate(np.array_split(np.arange(grid.shape[0]), up)):
        for j, horizontal in enumerate(np.array_split(np.arange(grid.shape[1]), across)):
            part = labels[np.ix_(vertical, horizontal)]
            summary = _area_summary(part, grid.solid_angle_per_cell)
            rows.append({
                "band": i + 1, "sector": j + 1,
                "bearing_start": number(bearings[horizontal[0]] % 360, 4),
                "bearing_end": number(bearings[horizontal[-1] + 1] % 360, 4),
                "elevation_start": number(altitudes[vertical[0]], 4),
                "elevation_end": number(altitudes[vertical[-1] + 1], 4),
                "open_determined": summary["percent_open_of_decisive"],
                "open_minimum": summary["bounds_open_percent_of_full_footprint"][0],
                "open_maximum": summary["bounds_open_percent_of_full_footprint"][1],
                "measured_percent": summary["percent_camera_support"],
                "decisive_percent": summary["percent_decisive"],
            })
    return rows


def angular_share(bench: Bench) -> Result:
    source = bench.one()
    data = bench.order.settings
    fields(data, "sectors", "bands")
    across = whole(data.get("sectors", 8), "Summary sectors", 1, 24)
    up = whole(data.get("bands", 4), "Summary elevation bands", 1, 12)
    grid = bench.grid
    samples = source.sample(grid)
    summary = _area_summary(samples.codes, grid.solid_angle_per_cell)
    rows = _sector_rows(bench, samples.codes, across, up)
    result = Result("open-share", "Open sky by solid angle",
                    "The open fraction mentions its denominator. Missing and ambiguous solid angle widen the stated bounds.")
    result.metrics = {**summary, "sampling": grid.describe()}
    result.readings = rows
    result.png("open-class-map", "Equal solid angle sky map", category_map(samples.codes, grid, "Sky by solid angle"))
    result.csv("angular-sectors", "Solid angle sector table", rows)
    result.warnings = source.method_warnings()
    result.probe_kind = "angular-cell"
    return result


def _horizon_samples(source, footprint: Footprint, count: int, height: int):
    bearings = footprint.bearing + np.arange(count) * footprint.sweep / count + footprint.sweep / (2 * count)
    elevations = np.linspace(footprint.bottom, footprint.top, height)
    az, el = np.meshgrid(bearings % 360, elevations)
    directions = np.stack((az, el), axis=-1)
    mapped = source.sample(directions)
    return bearings, elevations, mapped


def _horizon_rows(bearings: np.ndarray, elevations: np.ndarray, codes: np.ndarray) -> list[dict]:
    output = []
    for col, angle in enumerate(bearings):
        band = codes[:, col]
        blocked = np.flatnonzero(band == OCCLUDED)
        open_sky = np.flatnonzero(band == OPEN)
        highest = blocked[-1] if len(blocked) else None
        if highest is None:
            upper_open = open_sky
        else:
            upper_open = open_sky[open_sky > highest]
        first_open = upper_open[0] if len(upper_open) else None
        lower = elevations[highest] if highest is not None else None
        upper = elevations[first_open] if first_open is not None else None
        usable = np.count_nonzero((band == OPEN) | (band == OCCLUDED))
        if lower is None or upper is None:
            status = "edge_unresolved"
        elif first_open > highest + 1:
            status = "bounded_with_gap"
        else:
            status = "direct_bracket"
        output.append({
            "bearing": number(angle % 360, 4),
            "highest_obstruction": number(lower, 4),
            "next_confirmed_sky": number(upper, 4),
            "state": status,
            "decisive_share": number(usable / len(band)),
            "open_below_obstruction": int(np.count_nonzero(open_sky < highest)) if highest is not None else 0,
        })
    return output


def horizon_line(bench: Bench) -> Result:
    source = bench.one()
    data = bench.order.settings
    fields(data, "bearings", "elevations")
    slices = whole(data.get("bearings", 144), "Bearings", 16, 360)
    height = whole(data.get("elevations", 91), "Elevations", 20, 181)
    bearings, elevations, details = _horizon_samples(source, bench.order.footprint, slices, height)
    rows = _horizon_rows(bearings, elevations, details.codes)
    report = Result("horizon", "Obstruction skyline",
                    "For every bearing, the highest measured obstruction and the next confirmed sky form a bracket where supported.")
    report.metrics = {
        "bearings_checked": slices,
        "with_confirmed_bracket": sum(r["state"] != "edge_unresolved" for r in rows),
        "open_below_upper_obstruction": sum(r["open_below_obstruction"] > 0 for r in rows),
        "nearby_bearing_spacing_deg": number(bench.order.footprint.sweep / slices),
        "vertical_sample_spacing_deg": number((elevations[-1] - elevations[0]) / (height - 1)),
    }
    report.readings = rows
    report.png("skyline", "Skyline bounds", height_chart([
        {"name": "Highest visible obstruction", "color": GOLD, "values": [r["highest_obstruction"] for r in rows]},
        {"name": "Next visible sky", "color": MINT, "values": [r["next_confirmed_sky"] for r in rows]}
    ], "Measured skyline", "Elevation in degrees", "Bearings across the selected sweep", hard_bounds=(0, 90)))
    report.csv("skyline-readings", "Bearing evidence", rows)
    report.warnings = source.method_warnings()
    report.probe_kind = "direction"
    return report
