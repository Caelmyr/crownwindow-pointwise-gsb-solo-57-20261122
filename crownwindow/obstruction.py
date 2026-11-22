"""Manual obstruction typing: confirmed foliage, temporary cloth and unresolved patches.

An :class:`ObstructionLayer` records only what an operator can actually read on the
native photograph.  Ambiguous grey edges stay *unresolved* instead of being forced
into a colour class, and pixels hidden behind drawn polygons are never described as
visible sky.  All area numbers come from native camera pixels, including transparent
or otherwise unsupported pixels, so the coverage behind the marks stays auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import numpy as np
from PIL import Image, ImageDraw

from .camera import Camera
from .contracts import Problem, enum, fields, items, mapping, number, point, text
from .visual import DARK, PALE, _draw_font, photo_canvas


FOLIAGE = "foliage"
CLOTH = "cloth"
UNRESOLVED = "unresolved"
OBSTRUCTION_KINDS = (FOLIAGE, CLOTH, UNRESOLVED)
MAX_PATCHES = 120
MIN_VERTICES = 3
MAX_VERTICES = 60
MIN_UNIT_AREA = 0.000035

# Distinct, colour-blind-safe class tints used both on server previews and in CSS.
KIND_COLORS = {
    FOLIAGE: (74, 132, 68),
    CLOTH: (176, 112, 58),
    UNRESOLVED: (211, 178, 74),
}
KIND_LABELS = {
    FOLIAGE: "Foliage and branches",
    CLOTH: "Temporary shade cloth",
    UNRESOLVED: "Cannot separate",
}
PREVIEW_EDGE = 920


@dataclass(frozen=True)
class ObstructionPatch:
    """One operator-confirmed polygon in unit image coordinates."""

    label: str
    kind: str
    vertices: tuple[tuple[float, float], ...]
    note: str

    @classmethod
    def parse(cls, raw: dict, index: int) -> "ObstructionPatch":
        data = mapping(raw, "Obstruction patch")
        fields(data, "label", "kind", "vertices", "note")
        kind = enum(data.get("kind"), "Obstruction class", OBSTRUCTION_KINDS)
        vertices = tuple(point(value, "Patch vertex") for value in
                         items(data.get("vertices"), "Patch vertices", MIN_VERTICES, MAX_VERTICES))
        if _polygon_area(vertices) < MIN_UNIT_AREA:
            raise Problem(f"Obstruction patch {index + 1} is too small to confirm.")
        label = text(data.get("label") or f"{KIND_LABELS[kind]} {index + 1}",
                     "Patch label", 70)
        note = text(data.get("note"), "Patch note", 350, default="")
        return cls(label, kind, vertices, note)

    def public(self) -> dict:
        return asdict(self)

    def mask(self, camera: Camera) -> np.ndarray:
        canvas = Image.new("1", (camera.width, camera.height))
        points = [tuple(round(v) for v in xy) for xy in camera.unit_to_pixel(np.asarray(self.vertices))]
        ImageDraw.Draw(canvas).polygon(points, fill=1)
        return np.asarray(canvas, dtype=bool)


@dataclass(frozen=True)
class ObstructionLayer:
    patches: tuple[ObstructionPatch, ...]

    @classmethod
    def parse(cls, raw: dict | None) -> "ObstructionLayer":
        doc = mapping(raw or {}, "Obstruction layer")
        fields(doc, "patches")
        rows = items(doc.get("patches", []), "Obstruction patches", 0, MAX_PATCHES)
        return cls(tuple(ObstructionPatch.parse(row, i) for i, row in enumerate(rows)))

    def public(self) -> dict:
        return {"patches": [patch.public() for patch in self.patches]}


def _polygon_area(vertices: tuple[tuple[float, float], ...]) -> float:
    return abs(sum(a * d - c * b for (a, b), (c, d) in
                   zip(vertices, vertices[1:] + vertices[:1], strict=True))) / 2


def _mask_grid(camera: Camera, layer: ObstructionLayer) -> tuple[np.ndarray, list[np.ndarray]]:
    """Class-label grid (0=unmarked) and per-patch native masks in document order."""
    labels = np.zeros((camera.height, camera.width), dtype=np.uint8)
    masks: list[np.ndarray] = []
    code_by_kind = {kind: code for code, kind in enumerate(OBSTRUCTION_KINDS, start=1)}
    for patch in layer.patches:
        mask = patch.mask(camera)
        masks.append(mask)
        labels[mask & (labels == 0)] = code_by_kind[patch.kind]
    return labels, masks


def obstruction_stats(camera: Camera, layer: ObstructionLayer) -> dict:
    """Native pixel shares; unsupported and unmarked pixels stay explicit."""
    labels, masks = _mask_grid(camera, layer)
    total = camera.height * camera.width
    supported = int(camera.valid.sum())
    unsupported = total - supported
    marked = labels > 0
    marked_count = int(marked.sum())
    overlap = 0
    for mask in masks:
        overlap += int(mask.sum())
    overlap_pixels = overlap - marked_count
    rows = []
    class_counts = {kind: int((labels == code).sum())
                    for code, kind in enumerate(OBSTRUCTION_KINDS, start=1)}
    for patch, mask in zip(layer.patches, masks, strict=True):
        pixels = int(mask.sum())
        rows.append({
            "label": patch.label,
            "class": patch.kind,
            "class_label": KIND_LABELS[patch.kind],
            "native_pixels": pixels,
            "share_of_image_percent": number(100 * pixels / total, 4),
            "supported_pixels_inside": int((mask & camera.valid).sum()),
            "unsupported_pixels_inside": int((mask & ~camera.valid).sum()),
            "vertices": len(patch.vertices),
            "note": patch.note,
        })
    for index, row in enumerate(rows, start=1):
        row["index"] = index
    unmarked_supported = int(((labels == 0) & camera.valid).sum())
    summary = {
        "image_native_pixels": total,
        "supported_pixels": supported,
        "unsupported_pixels": unsupported,
        "unsupported_share_percent": number(100 * unsupported / total, 4),
        "marked_pixels": marked_count,
        "unmarked_supported_pixels": unmarked_supported,
        "overlapping_pixels": overlap_pixels,
        "shares": {
            kind: {
                "label": KIND_LABELS[kind],
                "native_pixels": class_counts[kind],
                "share_of_image_percent": number(100 * class_counts[kind] / total, 4),
                "share_of_marked_percent":
                    number(100 * class_counts[kind] / marked_count, 4) if marked_count else None,
            }
            for kind in OBSTRUCTION_KINDS
        },
    }
    summary["confirmed_obstruction_pixels"] = class_counts[FOLIAGE] + class_counts[CLOTH]
    summary["confirmed_share_of_image_percent"] = number(
        100 * summary["confirmed_obstruction_pixels"] / total, 4)
    summary["unresolved_share_of_image_percent"] = summary["shares"][UNRESOLVED]["share_of_image_percent"]
    warnings = []
    if overlap_pixels:
        warnings.append(
            f"{overlap_pixels} pixels fall inside more than one patch; each pixel is counted once, "
            "in the first patch drawn over it.")
    hidden_unsupported = int((marked & ~camera.valid).sum())
    if hidden_unsupported:
        warnings.append(
            f"{hidden_unsupported} drawn-over pixels have no measured camera support. They are recorded "
            "as hidden, never as visible sky.")
    return {"summary": summary, "patches": rows, "warnings": warnings}


def typing_overlay(camera: Camera, layer: ObstructionLayer, stats: dict, *, edge: int = PREVIEW_EDGE) -> Image.Image:
    """Original photograph with the operator's typed polygons drawn over it."""
    base = photo_canvas(camera, edge).convert("RGBA")
    width, height = base.size
    tint = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    tint_draw = ImageDraw.Draw(tint)
    outline = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    line_draw = ImageDraw.Draw(outline)
    for patch in layer.patches:
        points = [(u * (width - 1), v * (height - 1)) for u, v in patch.vertices]
        color = KIND_COLORS[patch.kind]
        tint_draw.polygon(points, fill=(*color, 104))
        line_draw.line(points + [points[0]], fill=(*color, 255), width=3, joint="curve")
        anchor = points[0]
        line_draw.ellipse((anchor[0] - 4, anchor[1] - 4, anchor[0] + 4, anchor[1] + 4),
                          fill=(*color, 255), outline=(255, 255, 255, 255))
    composed = Image.alpha_composite(base, tint)
    composed = Image.alpha_composite(composed, outline).convert("RGB")
    legend_height = 54
    chart = Image.new("RGB", (width, height + legend_height), DARK)
    chart.paste(composed, (0, 0))
    draw = ImageDraw.Draw(chart)
    x = 14
    for kind in OBSTRUCTION_KINDS:
        block = KIND_COLORS[kind]
        draw.rectangle((x, height + 17, x + 17, height + 34), fill=block)
        draw.text((x + 23, height + 16), KIND_LABELS[kind], fill=PALE, font=_draw_font(15))
        x += int(draw.textlength(KIND_LABELS[kind], font=_draw_font(15))) + 44
    marked = stats["summary"]["marked_pixels"]
    note = f"Marked {marked} of {stats['summary']['image_native_pixels']} native pixels · " \
           f"unmarked supported pixels stay unclaimed"
    draw.text((14, height + 36), note, fill=(172, 190, 184), font=_draw_font(13))
    return chart


def typing_csv(stats: dict) -> bytes:
    columns = ["index", "label", "class", "class_label", "native_pixels", "share_of_image_percent",
               "supported_pixels_inside", "unsupported_pixels_inside", "vertices", "note"]
    lines = [",".join(columns)]
    for row in stats["patches"]:
        lines.append(",".join(_csv_cell(row.get(column)) for column in columns))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _csv_cell(value) -> str:
    if value is None:
        return ""
    text_value = str(value)
    if any(marker in text_value for marker in (",", '"', "\n")):
        text_value = '"' + text_value.replace('"', '""') + '"'
    return text_value


def typing_json(camera: Camera, layer: ObstructionLayer, stats: dict) -> bytes:
    document = {
        "image": camera.public(),
        "classes": {kind: {"label": KIND_LABELS[kind], "color": list(KIND_COLORS[kind])}
                    for kind in OBSTRUCTION_KINDS},
        "layer": layer.public(),
        "statistics": stats,
    }
    return json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def preview_bundle(camera: Camera, layer: ObstructionLayer) -> dict:
    stats = obstruction_stats(camera, layer)
    overlay = typing_overlay(camera, layer, stats)
    return {
        "stats": stats,
        "overlay": overlay,
        "csv": typing_csv(stats),
        "json": typing_json(camera, layer, stats),
    }
