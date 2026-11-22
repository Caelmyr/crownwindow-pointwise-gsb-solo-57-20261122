"""Lens-family and reference-color diagnostics for prepared surveys."""

from __future__ import annotations

import numpy as np

from .bench import Bench, Source
from .calibration import Landmark, fit_lens
from .camera import describe_rgb, rgb_luminance, srgb_to_linear
from .contracts import Problem, angle, fields, items, mapping, number, real, text
from .projection import MODELS, Lens, directions_to_vectors, vectors_to_directions
from .result import Result
from .visual import GOLD, MINT, PINK, border_color_at, comparison_panel, height_chart, plot_over_camera


def _radial_model(source: Source, model: str) -> Lens:
    base = source.lens
    return Lens(base.center_uv, base.radius_over_short_side, base.maximum_zenith,
                base.north_clockwise, model)


def _outlier_pressure(rows: list[dict], percentile: float) -> dict:
    values = np.array([entry["residual_px"] for entry in rows])
    limit = float(np.quantile(values, percentile))
    worst = values.argmax()
    near = values[np.argsort(values)[:max(1, int(len(values) * .75))]]
    return {"worst_mark": rows[worst]["mark"], "worst_px": number(values[worst], 4),
            "percentile_px": number(limit, 4), "trimmed_rms_px": number(np.sqrt((near ** 2).mean()), 4)}


def _fit_families(source: Source, raw: list, target: float) -> tuple[list[dict], list[dict]]:
    rows = []
    detail = []
    for model in MODELS:
        tentative = _radial_model(source, model)
        try:
            result = fit_lens(source.camera, tentative, raw)
        except Problem as error:
            rows.append({"projection": model, "status": "not_constrained", "message": str(error)})
            continue
        earned = result.public()
        radii = np.array([90 - p.direction[1] for p in result.landmarks])
        spread = radii.max() - radii.min()
        rows.append({
            "projection": model,
            "status": "fitted" if spread >= 14 else "weak_radial_range",
            "rms_px": earned["metrics"]["rms_px"],
            "median_px": earned["metrics"]["median_px"],
            "landmark_radius_span_deg": number(spread),
            "fit_center_unit": list(result.lens.center_uv),
            "fit_radius_over_short_side": number(result.lens.radius_over_short_side, 6),
            "fit_north": number(result.lens.north_clockwise, 5),
            **_outlier_pressure(earned["residuals"], target),
        })
        detail.append({"projection": model, "residuals": earned["residuals"], "lens": earned["lens"]})
    return rows, detail


def _family_overlay(source, detail):
    items = []
    for q, item in enumerate(detail[:4]):
        color = border_color_at(q)
        pairs = [(row["entered_image"][0], row["entered_image"][1],
                  f"{row['mark'][:13]} {row['residual_px']:.1f}", color)
                 for row in item["residuals"]]
        items.append((item["projection"], plot_over_camera(source.camera, pairs, edge=620)))
    if not items:
        return plot_over_camera(source.camera, [])
    return comparison_panel(items, "Each tested radial law in the same source")


def _radial_rows(source, raw):
    results = []
    for i, row in enumerate(raw, 1):
        point = row.get("image")
        direction = angle(row.get("direction"))
        if point is None:
            raise Problem(f"Provide an image target for landmark {i}.")
        x, y = source.camera.unit_to_pixel(point)
        origin = source.camera.unit_to_pixel(source.lens.center_uv)
        actual = float(np.hypot(x - origin[0], y - origin[1]))
        expected = float(source.lens.radial_fraction(90 - direction[1]) * source.lens.scale_px(source.camera))
        results.append({"mark": row.get("title", f"Mark {i}"), "measured_px": number(actual, 3),
                        "predicted_px_current_lens": number(expected, 3),
                        "radial_difference_px": number(actual - expected, 3), "zenith_distance": number(90 - direction[1], 3)})
    return results


DEMO_OPTIC_MARKS = [
    {"title": "Near north", "image": [.53, .305], "direction": [1., 53.]},
    {"title": "East branch", "image": [.825, .535], "direction": [90., 27.]},
    {"title": "South bend", "image": [.47, .845], "direction": [180., 26.]},
    {"title": "Left skyline", "image": [.14, .56], "direction": [258., 23.]},
    {"title": "Inner west", "image": [.37, .49], "direction": [274., 67.]},
    {"title": "Upper branch", "image": [.578, .377], "direction": [35., 59.]},
]


def lens_model_audit(bench: Bench) -> Result:
    source = bench.one()
    options = bench.order.settings
    fields(options, "marks", "tail")
    raw = [mapping(row, "Survey landmark") for row in items(options.get("marks", DEMO_OPTIC_MARKS),
                                                            "Lens landmarks", 5, 24)]
    raw = [Landmark.parse(mark, index).public() for index, mark in enumerate(raw)]
    percentile = real(options.get("tail", .85), "Residual percentile", .65, .99)
    rows, detail = _fit_families(source, raw, percentile)
    solved = [row for row in rows if row["status"] in ("fitted", "weak_radial_range")]
    ranked = sorted(solved, key=lambda row: row["rms_px"])
    summary = {"families_checked": len(rows), "mark_count": len(raw),
               "lowest_rms": ranked[0]["projection"] if ranked else None,
               "separation_from_next_rms_px": number(ranked[1]["rms_px"] - ranked[0]["rms_px"]) if len(ranked) > 1 else None}
    radial = _radial_rows(source, raw)
    report = Result("lens-audit", "Radial law residual check",
                    "The same native landmarks are checked under different level-camera lens laws. The current lens remains selected.")
    report.metrics = summary
    report.readings = rows
    report.png("radial-comparison", "Independent radial fit residuals", _family_overlay(source, detail))
    lines = [
        {"name": row["projection"], "color": border_color_at(i),
         "values": [x["residual_px"] for x in row["residuals"]]} for i, row in enumerate(detail)
    ]
    if lines:
        report.png("radial-residual-chart", "Residuals by entered landmark", height_chart(
            lines, "Angular lens fits", "Native error in pixels", "Survey landmarks in supplied order"))
    report.csv("radial-models", "Lens family comparison", rows)
    report.csv("current-lens-radii", "Radii under the retained lens", radial)
    report.warnings = ["Small residuals are not evidence of performance outside the marked radial range."]
    report.probe_kind = "image"
    return report


def _disk_directions(center: tuple[float, float], radius: float, steps: int):
    bearing, elevation = center
    if elevation <= 1e-7 and radius > 0:
        elevation = 1e-7
    x = np.linspace(-1, 1, steps) * radius
    y = np.linspace(-1, 1, steps) * radius
    gx, gy = np.meshgrid(x, y)
    disc = gx ** 2 + gy ** 2 <= radius ** 2
    horizontal = gx[disc]
    vertical = gy[disc]
    v0 = directions_to_vectors(np.array([bearing, elevation]))
    unit_east = directions_to_vectors(np.array([bearing + 90, 0]))
    unit_up = np.cross(unit_east, v0)
    ramp = np.radians(np.hypot(horizontal, vertical))
    px = np.divide(horizontal, np.hypot(horizontal, vertical), out=np.zeros_like(horizontal),
                   where=np.hypot(horizontal, vertical) > .000001)
    py = np.divide(vertical, np.hypot(horizontal, vertical), out=np.zeros_like(vertical),
                   where=np.hypot(horizontal, vertical) > .000001)
    tangent = px[:, None] * unit_east + py[:, None] * unit_up
    disk = np.cos(ramp)[:, None] * v0 + np.sin(ramp)[:, None] * tangent
    return vectors_to_directions(disk)


def _luminance_patches(first: Source, second: Source, name: str, center, radius: float):
    common = _disk_directions(center, radius, 17)
    a, b = first.sample(common), second.sample(common)
    sampled = a.measured & b.measured
    sky = a.known & b.known
    measured_common = sampled & sky
    n = int(np.count_nonzero(measured_common))
    row = {"reference": name, "center_direction": list(center), "rays_checked": len(common),
           "common_camera_rays": int(sampled.sum()), "joint_decisive_rays": n,
           "class_disagreement": int(np.count_nonzero(measured_common & (a.codes != b.codes)))}
    if n < 6:
        row.update(luminance_first=None, luminance_second=None,
                   per_ray_ratio_p10=None, median_camera_ratio=None, per_ray_ratio_p90=None)
        return row
    rgb_a = a.rgb[measured_common]
    rgb_b = b.rgb[measured_common]
    l1 = rgb_luminance(srgb_to_linear(rgb_a))
    l2 = rgb_luminance(srgb_to_linear(rgb_b))
    ratio = l2[l1 > .02] / l1[l1 > .02]
    spread = np.quantile(ratio, [.10, .5, .9]) if len(ratio) else [None] * 3
    row.update(
        rgb_first=describe_rgb(rgb_a)["median"], rgb_second=describe_rgb(rgb_b)["median"],
        luminance_first=number(np.median(l1)), luminance_second=number(np.median(l2)),
        per_ray_ratio_p10=number(spread[0]), median_camera_ratio=number(spread[1]), per_ray_ratio_p90=number(spread[2]),
    )
    return row


def _read_tones(raw: list) -> list[tuple[str, tuple[float, float], float]]:
    patches = []
    for i, entry in enumerate(items(raw, "Direction patches", 1, 20)):
        data = mapping(entry, "Direction patch")
        fields(data, "title", "direction", "radius")
        name = text(data.get("title") or f"Direction {i + 1}", "Patch name", 60)
        center = angle(data.get("direction"), "Patch center")
        radius = real(data.get("radius", 3), "Patch radius", .5, 15)
        patches.append((name, center, radius))
    return patches


def _reference_overlay(first, second, samples):
    views = []
    for s in (first, second):
        direction = np.array([x[1] for x in samples])
        points, geometric = s.lens.world_to_image(direction, s.camera)
        marked = [(u, v, samples[i][0], border_color_at(i)) for i, ((u, v), seen) in
                  enumerate(zip(points, geometric, strict=True)) if seen]
        views.append((s.title, plot_over_camera(s.camera, marked)))
    return comparison_panel(views, "Repeat brightness reference directions")


DEMO_BRIGHTNESS = [
    {"title": "Upper clearing", "direction": [55, 68], "radius": 4},
    {"title": "West sky", "direction": [268, 48], "radius": 4},
    {"title": "Low branches", "direction": [135, 18], "radius": 3},
]


def relative_light_audit(bench: Bench) -> Result:
    first, second = bench.two()
    settings = bench.order.settings
    fields(settings, "patches")
    sampled = _read_tones(settings.get("patches", DEMO_BRIGHTNESS))
    rows = [_luminance_patches(first, second, name, direction, radius) for name, direction, radius in sampled]
    chart = height_chart([
        {"name": "Median camera ratio", "color": MINT, "values": [row["median_camera_ratio"] for row in rows]},
        {"name": "Low within patch", "color": GOLD, "values": [row["per_ray_ratio_p10"] for row in rows]},
        {"name": "High within patch", "color": PINK, "values": [row["per_ray_ratio_p90"] for row in rows]},
    ], "Brightness agreement under repeat capture", "Second / first", "Selected direction patch", reference=1)
    report = Result("exposure-audit", "Repeat brightness reference check",
                    "Compare repeated camera codes on directions supported on both photographs without renormalizing those photographs.")
    report.metrics = {"evaluated_patches": len(rows), "usable_patches": sum(r["median_camera_ratio"] is not None for r in rows),
                      "paired_camera_rays": sum(row["common_camera_rays"] for row in rows),
                      "with_class_disagreement": sum(row["class_disagreement"] > 0 for row in rows)}
    report.readings = rows
    report.png("brightness-reference-positions", "Same patch centers in source photographs",
               _reference_overlay(first, second, sampled))
    report.png("reference-ratio", "Per-patch camera ratio bounds", chart)
    report.csv("reference-measurements", "Repeat measurements with common support", rows)
    report.warnings = ["Camera values are exposure dependent. This comparison reports differences and does not apply a correction."]
    report.probe_kind = "direction"
    return report
