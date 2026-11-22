"""Native field quality and sampled color-model checks."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .bench import Bench
from .camera import rgb_luminance, srgb_to_linear
from .contracts import Problem, fields, items, number, real, whole
from .result import Result
from .sky import Classifier, Controls, LABELS, MISSING, OCCLUDED, OPEN, Region, UNCERTAIN
from .visual import GOLD, MINT, PINK, _draw_font, normalized_heat, photo_canvas


def _image_resolution(camera, bound: int):
    scale = min(1., bound / max(camera.color.shape[:2]))
    if scale >= 1.0:
        return camera.color, camera.valid, 1.
    shape = (round(camera.width * scale), round(camera.height * scale))
    rgb = Image.fromarray(np.rint(camera.color * 255).astype(np.uint8))
    rgb = np.asarray(rgb.resize(shape, resample=Image.Resampling.BILINEAR), dtype=np.float32) / 255
    valid = Image.fromarray(camera.valid.astype(float).astype(np.float32))
    valid = np.asarray(valid.resize(shape, resample=Image.Resampling.BILINEAR)) >= .998
    return rgb, valid, scale


def _gradients(values: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A derivative estimate never crosses native missing pixels."""
    working = np.where(support, values, 0)
    small = ndimage.gaussian_filter(working, .7, mode="reflect")
    grad_x = ndimage.sobel(small, axis=1, mode="nearest") / 8
    grad_y = ndimage.sobel(small, axis=0, mode="nearest") / 8
    grad_support = ndimage.binary_erosion(support, structure=np.ones((7, 7)), border_value=0)
    return grad_x, grad_y, grad_support


def _measure_tile(luma: np.ndarray, grad_x: np.ndarray, grad_y: np.ndarray, support: np.ndarray,
                  source_support: np.ndarray, native_bounds: list[float], coords: tuple[int, int],
                  signal_cutoff: float) -> dict:
    row, column = coords
    values = luma[source_support]
    usable = int(support.sum())
    result = {"row": row + 1, "column": column + 1, "native_bounds": native_bounds,
              "declared_share": number(len(values) / luma.size), "gradient_share": number(usable / luma.size)}
    if usable < luma.size * .25:
        return {**result, "state": "insufficient_pixels", "edge_p95": None,
                "direction_concentration": None, "axis_degrees_from_x": None, "linear_luminance_range": None}
    signal = np.hypot(grad_x[support], grad_y[support])
    p95 = np.quantile(signal, .95)
    q10, q90 = np.quantile(values, [.10, .90])
    if p95 < signal_cutoff:
        return {**result, "state": "low_contrast", "edge_p95": number(p95),
                "direction_concentration": None, "axis_degrees_from_x": None, "linear_luminance_range": number(q90 - q10)}
    strong = support & (np.hypot(grad_x, grad_y) >= max(signal_cutoff, p95 / 3))
    xx = np.mean(grad_x[strong] ** 2)
    yy = np.mean(grad_y[strong] ** 2)
    xy = np.mean(grad_x[strong] * grad_y[strong])
    angle = np.degrees(np.arctan2(2 * xy, xx - yy) * .5) % 180
    coherence = np.hypot(xx - yy, 2 * xy) / max(1e-9, xx + yy)
    return {**result, "state": "has_detail", "edge_p95": number(p95),
            "direction_concentration": number(coherence), "axis_degrees_from_x": number(angle, 4),
            "linear_luminance_range": number(q90 - q10)}


def _blocks(width, height, nx, ny):
    x_edges = np.linspace(0, width, nx + 1).round().astype(int)
    y_edges = np.linspace(0, height, ny + 1).round().astype(int)
    for iy, (y0, y1) in enumerate(zip(y_edges[:-1], y_edges[1:], strict=True)):
        for ix, (x0, x1) in enumerate(zip(x_edges[:-1], x_edges[1:], strict=True)):
            yield iy, ix, (int(x0), int(y0), int(x1), int(y1))


def _detail_grid(camera, nx, ny, cutoff):
    rgb, measured, factor = _image_resolution(camera, 1200)
    y = rgb_luminance(srgb_to_linear(rgb))
    gx, gy, calculations = _gradients(y, measured)
    rows = []
    for i, j, (x0, y0, x1, y1) in _blocks(y.shape[1], y.shape[0], nx, ny):
        box = np.s_[y0:y1, x0:x1]
        native = [round(x0 / factor), round(y0 / factor), min(camera.width, round(x1 / factor)),
                  min(camera.height, round(y1 / factor))]
        rows.append(_measure_tile(y[box], gx[box], gy[box], calculations[box],
                                  measured[box], native, (i, j), cutoff))
    return rows, factor


def _detail_map(camera, rows):
    preview = photo_canvas(camera, 885)
    draw = ImageDraw.Draw(preview)
    scale = [preview.width / camera.width, preview.height / camera.height]
    colors = {"insufficient_pixels": (145, 157, 164), "low_contrast": GOLD, "has_detail": MINT}
    for row in rows:
        x0, y0, x1, y1 = (value * scale[i % 2] for i, value in enumerate(row["native_bounds"]))
        style = colors[row["state"]]
        draw.rectangle((x0 + 1, y0 + 1, max(x0 + 1, x1 - 1), max(y0 + 1, y1 - 1)), outline=style, width=1)
        if x1 - x0 >= 44 and y1 - y0 >= 26:
            val = f"{row['edge_p95']:.3f}" if row["edge_p95"] is not None else "—"
            draw.text((x0 + 4, y0 + 3), val, font=_draw_font(13), fill=style,
                      stroke_width=2, stroke_fill=(22, 30, 31))
    return preview


def field_detail(bench: Bench) -> Result:
    source = bench.one()
    data = bench.order.settings
    fields(data, "columns", "rows", "edge_floor")
    nx = whole(data.get("columns", 8), "Columns", 3, 22)
    ny = whole(data.get("rows", 6), "Rows", 3, 18)
    cutoff = real(data.get("edge_floor", .023), "Resolvable edge level", .001, .25)
    rows, factor = _detail_grid(source.camera, nx, ny, cutoff)
    positive = [value["edge_p95"] for value in rows if value["state"] == "has_detail"]
    report = Result("field-detail", "Field detail across the photograph",
                    "The map locates blocks with visible detail without treating an empty or featureless patch as a focus error.")
    report.metrics = {"grid": [nx, ny], "working_pixels_per_native_pixel": number(factor),
                      "measurable_blocks": sum(row["state"] != "insufficient_pixels" for row in rows),
                      "strong_detail_blocks": len(positive),
                      "median_edge_level_in_detail": number(np.median(positive)) if positive else None,
                      "edge_level_floor": cutoff}
    report.readings = rows
    report.png("native-detail", "Native image evidence by block", _detail_map(source.camera, rows))
    value = np.array([item["edge_p95"] or 0 for item in rows]).reshape(ny, nx)
    valid = np.array([item["state"] != "insufficient_pixels" for item in rows]).reshape(ny, nx)
    heat = Image.fromarray(normalized_heat(value, valid, 0, max(cutoff * 3, .06))).resize((880, 600), Image.Resampling.NEAREST)
    report.png("edge-score-grid", "Observed local gradient levels", heat)
    report.csv("image-detail-cells", "Native block quality readings", rows)
    report.warnings = ["Strong edges indicate image detail. Textureless sky and uniform branches give no independent focus measurement."]
    report.probe_kind = "image"
    return report


def _holdout_samples(camera, lens, region: Region):
    native = region.mask(camera) & camera.valid
    yy, xx = np.nonzero(native)
    skip = max(1, len(xx) // 28000)
    yy, xx = yy[::skip], xx[::skip]
    unit = camera.pixel_to_unit(np.stack((xx, yy), axis=-1))
    _, in_lens = lens.image_to_world(unit, camera)
    values = camera.color[yy, xx]
    return unit, values, in_lens


def _validation_row(region: Region, classifier: Classifier, camera, lens):
    unit, values, photograph = _holdout_samples(camera, lens, region)
    if photograph.sum() < 12:
        raise Problem(f"Validation region {region.label} has too few measured pixels inside the lens.")
    categories, margin = classifier.classify(values, photograph, unit)
    expected = OPEN if region.kind == "sky" else OCCLUDED
    decisive = np.isin(categories, [OPEN, OCCLUDED])
    agrees = categories == expected
    false = decisive & ~agrees
    return {
        "region": region.label, "expected": LABELS[expected], "pixels_checked": len(unit),
        "pixels_inside_lens": int(photograph.sum()),
        "confirmed_agreement": int(agrees.sum()), "opposite_class": int(false.sum()),
        "ambiguous": int(np.count_nonzero(categories == UNCERTAIN)),
        "withheld": int(np.count_nonzero(categories == MISSING)),
        "median_sky_score": number(np.nanmedian(margin)) if np.isfinite(margin).any() else None,
    }


def _validation_preview(camera, regions, records):
    preview = photo_canvas(camera)
    draw = ImageDraw.Draw(preview)
    for index, (region, result) in enumerate(zip(regions, records, strict=True)):
        points = [(u * preview.width, v * preview.height) for u, v in region.vertices]
        issue = result["opposite_class"] > result["confirmed_agreement"]
        style = PINK if issue else MINT
        draw.line(points + points[:1], fill=style, width=3)
        draw.text(points[0], str(index + 1) + " · " + region.label, fill=style,
                  stroke_width=2, stroke_fill=(27, 43, 41), font=_draw_font(14))
    return preview


def check_samples(bench: Bench) -> Result:
    source = bench.one()
    data = bench.order.settings
    fields(data, "regions")
    inside = items(data.get("regions", [
        {"label": "Center sky", "kind": "sky", "vertices": [[.43, .43], [.55, .43], [.55, .55], [.43, .55]]}
    ]), "Validation regions", 1, 18)
    regions = [Region.parse(raw, i) for i, raw in enumerate(inside)]
    if any(r.kind == "exclude" for r in regions):
        raise Problem("Use labelled sky or obstruction regions to check the classifier.")
    if any(region.vertices == training.vertices for region in regions for training in source.controls.regions):
        raise Problem("The exact training patch is already part of the model. Check another location.")
    rows = [_validation_row(region, source.classifier, source.camera, source.lens) for region in regions]
    report = Result("color-check", "Independent sky sample check",
                    "Check an already prepared classifier on separate labelled regions without recalibrating the image.")
    report.metrics = {"regions": len(rows), "confirmed_agreement": sum(row["confirmed_agreement"] for row in rows),
                      "opposite_class": sum(row["opposite_class"] for row in rows),
                      "ambiguous": sum(row["ambiguous"] for row in rows),
                      "withheld": sum(row["withheld"] for row in rows)}
    report.readings = rows
    report.png("validation-patches", "Checked regions on the source", _validation_preview(source.camera, regions, rows))
    report.csv("patch-agreement", "Check region results", rows)
    report.warnings = source.method_warnings()
    report.probe_kind = "image"
    return report
