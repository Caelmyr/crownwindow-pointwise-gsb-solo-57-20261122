"""Directional field-mark readout and inspection of saved source bounds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bench import Bench, Source
from .camera import rgb_luminance, srgb_to_linear
from .contracts import Mark, Problem, angle, fields, items, number, point, real
from .projection import angular_distance, bearing_delta
from .result import Result
from .sky import LABELS, MISSING
from .visual import draw_marks


@dataclass(frozen=True)
class MarkReadout:
    mark: Mark
    position: np.ndarray | None
    direction: np.ndarray | None
    observed: bool
    distinction: str
    discrepancy: float | None
    rgb: np.ndarray | None

    def public(self, index: int) -> dict:
        position = [number(value, 6) for value in self.position] if self.position is not None else None
        return {
            "mark": index + 1,
            "title": self.mark.title,
            "mark_role": self.mark.role,
            "image_unit": position,
            "direction": [number(value, 5) for value in self.direction] if self.direction is not None else None,
            "declared_direction": list(self.mark.direction) if self.mark.direction is not None else None,
            "status": self.distinction,
            "measured_pixel": self.observed,
            "discrepancy_degrees": number(self.discrepancy),
            "rgb": [number(value, 5) for value in self.rgb] if self.rgb is not None else None,
        }


def read_mark(source: Source, mark: Mark, tolerance: float) -> MarkReadout:
    if mark.native is not None:
        observed = np.asarray(mark.native)
        direction, permitted = source.lens.image_to_world(observed, source.camera)
        direction = direction if permitted else None
        if mark.direction is not None and permitted:
            error = float(angular_distance(direction, mark.direction))
            classification = "both_in_agreement" if error <= tolerance else "bearing_disagrees"
        elif permitted:
            error = None
            classification = "native_mark"
        else:
            error = None
            classification = "out_of_lens"
    else:
        direction = np.asarray(mark.direction)
        observed, permitted = source.lens.world_to_image(direction, source.camera)
        if not permitted:
            return MarkReadout(mark, None, direction, False, "out_of_lens", None, None)
        classification = "direction_pin"
        error = None
    rgb, opaque = source.camera.bilinear(np.array([observed]))
    return MarkReadout(mark, observed, direction, bool(opaque[0]), classification,
                       error, rgb[0] if opaque[0] else None)


def field_overlay(bench: Bench) -> Result:
    source = bench.one()
    data = bench.order.settings
    fields(data, "marks", "tolerance", "origin")
    raw = items(data.get("marks", [{"title": "Survey north", "direction": [0, 25], "role": "reference"},
           {"title": "Upper canopy", "direction": [48, 68], "role": "opening"}]), "Field marks", 1, 90)
    marks = [Mark.parse(row) for row in raw]
    tolerance = real(data.get("tolerance", 3), "Direction agreement tolerance in degrees", .2, 45)
    origin = real(data.get("origin", 0), "Arrange from bearing", -360, 720)
    observed = [read_mark(source, mark, tolerance) for mark in marks]
    serialized = [row.public(i) for i, row in enumerate(observed)]
    rank = _sweep_order(serialized, origin)
    report = Result("field-marks", "Field landmarks on the lens",
                    "Compare marked landmarks with the direction recorded in the field, and preserve where each mark came from.")
    report.metrics = {"entries": len(marks), "with_native_support": sum(row.observed for row in observed),
                      "bearing_disagreements": sum(row.distinction == "bearing_disagrees" for row in observed),
                      "angular_tolerance_deg": tolerance, "bearing_order_origin": origin % 360}
    report.readings = rank
    render_marks = [Mark(
        row.mark.title,
        tuple(row.position) if row.position is not None else None,
        row.mark.direction,
        row.mark.role,
        row.mark.note,
    ).public() for row in observed if row.position is not None]
    report.png("field-on-camera", "Field markings on original capture", draw_marks(source.camera, source.lens, render_marks))
    report.csv("landmark-readings", "Degrees and native sample positions", rank)
    report.probe_kind = "image"
    return report


def _sweep_order(rows: list[dict], origin: float) -> list[dict]:
    chosen = []
    blank = []
    for row in rows:
        if row["direction"] is None:
            row["offset_from_origin"] = None
            blank.append(row)
        else:
            row["offset_from_origin"] = number((row["direction"][0] - origin) % 360)
            chosen.append(row)
    return sorted(chosen, key=lambda row: (row["offset_from_origin"], row["mark"])) + blank


def direction_readout(source: Source, bearings: list) -> list[dict]:
    data = [angle(entry) for entry in items(bearings, "Directions", 1, 80)]
    samples = source.sample(np.asarray(data))
    output = []
    for index in range(len(data)):
        report = samples.readout(index)
        if report["image_unit"] is not None:
            pixels = source.camera.unit_to_pixel(samples.native[index])
            report["camera_px"] = [number(value, 3) for value in pixels]
            report["linear_display_luminance"] = number(rgb_luminance(srgb_to_linear(samples.rgb[index])))
        else:
            report.update(camera_px=None, linear_display_luminance=None)
        report["source"] = source.public()
        output.append(report)
    return output


def image_readout(source: Source, coords: list, radius: float):
    positions = [point(entry) for entry in items(coords, "Image positions", 1, 80)]
    readings = []
    for p in positions:
        row = source.lens.readout(source.camera, p)
        row["nearby"] = source.camera.support_window(p, radius)
        if row["direction"] is not None and row["valid"]:
            values = np.asarray([p])
            color, support = source.camera.bilinear(values)
            code, margin = source.classifier.classify(color, support, values)
            row["class"] = LABELS[int(code[0])]
            row["sky_margin"] = number(margin[0])
        else:
            row["class"] = LABELS[MISSING]
            row["sky_margin"] = None
        readings.append(row)
    return readings


def native_witness(source: Source, readout: dict, radius: float):
    """Draw the exact camera neighbourhood used by a point probe."""
    from PIL import Image, ImageDraw

    from .visual import DARK, GOLD, MINT, MUTED, PALE, _draw_font

    uv = readout.get("image_unit") or readout.get("unit")
    card = Image.new("RGB", (690, 460), DARK)
    draw = ImageDraw.Draw(card)
    draw.text((20, 13), source.title[:61], fill=PALE, font=_draw_font(22))
    if uv is None:
        draw.text((22, 67), "No camera pixel supports this direction.", fill=GOLD, font=_draw_font(20))
        return card
    camera = source.camera
    ix, iy = camera.unit_to_pixel(uv).round().astype(int)
    radius = max(1, min(32, round(radius)))
    inset = max(5, radius)
    size = (2 * inset + 1)
    known = np.zeros((size, size), dtype=bool)
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:] = (52, 65, 74)
    left, top = max(0, ix - inset), max(0, iy - inset)
    right, bottom = min(camera.width, ix + inset + 1), min(camera.height, iy + inset + 1)
    shift_x, shift_y = left - ix + inset, top - iy + inset
    block = np.s_[shift_y:shift_y + bottom - top, shift_x:shift_x + right - left]
    source_support = camera.valid[top:bottom, left:right]
    known[block] = source_support
    patch = np.rint(camera.color[top:bottom, left:right] * 255).astype(np.uint8)
    rgb[block] = np.where(source_support[..., None], patch, rgb[block])
    factor = max(1, min(24, 350 // size))
    image = Image.fromarray(rgb).resize((size * factor, size * factor), Image.Resampling.NEAREST)
    card.paste(image, (20, 72))
    corner = (inset - radius) * factor
    outer = (inset + radius + 1) * factor
    draw.rectangle((20 + corner, 72 + corner, 20 + outer - 1, 72 + outer - 1), outline=MINT, width=2)
    x = 20 + (inset + .5) * factor
    y = 72 + (inset + .5) * factor
    draw.line((x - 8, y, x + 8, y), fill=GOLD, width=2)
    draw.line((x, y - 8, x, y + 8), fill=GOLD, width=2)
    read = {"image_pixel": [int(ix), int(iy)], "screen_radius": int(radius),
            "labeled_as": readout.get("class"), "evidence_found": bool(known[inset, inset]),
            "direction": readout.get("direction")}
    _witness_text(draw, read, left=407, top=84)
    return card


def _witness_text(draw, reading, left, top):
    from .visual import GOLD, MUTED, PALE, _draw_font

    bearing, elevation = reading["direction"] or (None, None)
    positions = [
        ("Native image pixel", str(tuple(reading["image_pixel"]))),
        ("Evidence", "Measured" if reading["evidence_found"] else "No measured pixel"),
        ("Classification", (reading["labeled_as"] or "unavailable").replace("_", " ")),
        ("Bearing / elevation", f"{bearing:.2f}° / {elevation:.2f}°" if bearing is not None else "Outside calibrated lens"),
        ("Checked sample radius", str(reading["screen_radius"]) + " native pixels"),
    ]
    for offset, (label, value) in enumerate(positions):
        y = top + offset * 57
        draw.text((left, y), label, font=_draw_font(14), fill=MUTED)
        draw.text((left, y + 21), value[:33], font=_draw_font(17), fill=PALE)
    draw.text((left, top + 302), "Gold marks the central camera pixel.", font=_draw_font(14), fill=GOLD)
