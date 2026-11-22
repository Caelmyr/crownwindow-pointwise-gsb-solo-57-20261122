"""Native window sampling with geographic coordinates and independent evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .bench import Bench, Source
from .camera import describe_rgb, rgb_luminance, srgb_to_linear
from .contracts import Problem, fields, items, mapping, number, point, text, whole
from .projection import angular_distance, directions_to_vectors, vectors_to_directions
from .result import Result
from .sky import LABELS, state_counts
from .visual import GOLD, PINK, _draw_font, comparison_panel, photo_canvas


@dataclass(frozen=True)
class Window:
    title: str
    a: tuple[float, float]
    b: tuple[float, float]

    @classmethod
    def parse(cls, raw: dict, index: int) -> Window:
        data = mapping(raw, "Source window")
        fields(data, "title", "corners")
        a, b = items(data.get("corners"), "Source window corners", 2, 2)
        a, b = point(a, "First corner"), point(b, "Second corner")
        if abs(a[0] - b[0]) < .007 or abs(a[1] - b[1]) < .007:
            raise Problem("Choose a window with measurable width and height.")
        return cls(text(data.get("title") or f"Window {index + 1}", "Window title", 60),
                   (min(a[0], b[0]), min(a[1], b[1])),
                   (max(a[0], b[0]), max(a[1], b[1])))

    def native_rect(self, source: Source) -> list[int]:
        left, top = source.camera.unit_to_pixel(self.a)
        right, bottom = source.camera.unit_to_pixel(self.b)
        return [int(np.ceil(left)), int(np.ceil(top)), int(np.floor(right)) + 1, int(np.floor(bottom)) + 1]


def _extract(source: Source, window: Window, grid_budget: int):
    camera = source.camera
    x0, y0, x1, y1 = window.native_rect(source)
    if min(x1 - x0, y1 - y0) < 2:
        raise Problem(f"{window.title} contains fewer than two complete pixels in either dimension.")
    data = camera.color[y0:y1, x0:x1]
    valid = camera.valid[y0:y1, x0:x1]
    step = max(1, int(np.sqrt((x1 - x0) * (y1 - y0) / grid_budget)))
    sample_y, sample_x = np.mgrid[y0:y1:step, x0:x1:step]
    unit = camera.pixel_to_unit(np.stack((sample_x, sample_y), axis=-1))
    clipped = data[::step, ::step]
    measured = valid[::step, ::step]
    direction, in_lens = source.lens.image_to_world(unit, camera)
    sampled, _ = source.classifier.classify(clipped, measured & in_lens, unit)
    return data, valid, step, direction, in_lens, measured, sampled


def _shape_report(direction: np.ndarray, valid: np.ndarray) -> dict:
    values = direction[valid]
    if not len(values):
        return {"angular_mean": None, "angular_spread_degrees": None,
                "min_elevation": None, "max_elevation": None}
    mean = directions_to_vectors(values).mean(axis=0)
    centre = vectors_to_directions(mean) if np.linalg.norm(mean) > 1e-6 else None
    errors = angular_distance(values, centre) if centre is not None else None
    return {"angular_mean": [number(v, 4) for v in centre] if centre is not None else None,
            "angular_spread_degrees": number(float(np.quantile(errors, .9)), 5) if errors is not None else None,
            "min_elevation": number(values[:, 1].min(), 4), "max_elevation": number(values[:, 1].max(), 4)}


def _patch_stats(source: Source, window: Window, budget: int):
    data, valid, step, direction, lens, measured, classification = _extract(source, window, budget)
    rgb = data[valid]
    if rgb.size:
        luminance = rgb_luminance(srgb_to_linear(rgb))
        tails = np.quantile(luminance, [.025, .5, .975])
        summary = [number(value, 5) for value in tails]
    else:
        summary = None
    source_summary = describe_rgb(rgb)
    geometry = _shape_report(direction, lens & measured)
    states = state_counts(classification)
    return {
        "window": window.title,
        "native_bounds": window.native_rect(source),
        "measured_count": int(valid.sum()),
        "source_pixels": int(valid.size),
        "fraction_measured": number(np.mean(valid), 6),
        "camera_RGB_median": source_summary["median"],
        "linear_luminance_interval": summary,
        "footprint_sample_step_px": step,
        "angular_class_samples": {name: states[name]["count"] for name in LABELS},
        **geometry,
    }


def _view_loupe(source, windows, selected):
    base = photo_canvas(source.camera)
    draw = ImageDraw.Draw(base)
    for index, window in enumerate(windows):
        a = window.a[0] * base.width, window.a[1] * base.height
        b = window.b[0] * base.width, window.b[1] * base.height
        color = GOLD if index == selected else PINK
        draw.rectangle((*a, *b), outline=color, width=3)
        draw.text((a[0] + 2, a[1] + 4), window.title[:30], fill=color,
                  stroke_width=2, stroke_fill=(28, 34, 34), font=_draw_font(16))
    camera = source.camera
    x0, y0, x1, y1 = windows[selected].native_rect(source)
    rgb = np.rint(camera.color[y0:y1, x0:x1] * 255).astype(np.uint8)
    visible = camera.valid[y0:y1, x0:x1]
    rgb[~visible] = (40, 53, 63)
    crop = Image.fromarray(rgb)
    return comparison_panel([("Original image and windows", base), (windows[selected].title, crop)],
                            "Native evidence window")


def native_window(bench: Bench) -> Result:
    source = bench.one()
    options = bench.order.settings
    fields(options, "windows", "detail_window", "angular_budget")
    windows = [Window.parse(doc, i) for i, doc in enumerate(items(options.get("windows", [
        {"title": "Inner canopy", "corners": [[.35, .35], [.62, .62]]}
    ]), "Inspection windows", 1, 16))]
    chosen = whole(options.get("detail_window", 1), "Enlarged window", 1, len(windows)) - 1
    budget = whole(options.get("angular_budget", 12000), "Direction samples per window", 200, 55000)
    rows = [_patch_stats(source, window, budget) for window in windows]
    report = Result("camera-window", "Native camera windows",
                    "Read camera values and directional coverage separately for each marked rectangular source patch.")
    report.metrics = {"windows": len(windows), "magnified_window": windows[chosen].title,
                      "supported_camera_pixels": sum(row["measured_count"] for row in rows),
                      "windows_with_angular_location": sum(row["angular_mean"] is not None for row in rows)}
    report.readings = rows
    report.png("camera-loupe", "Original window locations and selected detail", _view_loupe(source, windows, chosen))
    report.csv("native-window-readings", "Source pixel and angular readings", rows)
    report.warnings = source.method_warnings()
    report.probe_kind = "image"
    return report
