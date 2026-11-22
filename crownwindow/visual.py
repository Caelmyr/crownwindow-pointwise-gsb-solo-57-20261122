"""Readable camera evidence, spherical maps, and quantitative charts."""

from __future__ import annotations

import math
from typing import Iterable
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .camera import Camera, bounded_size
from .contracts import Palette
from .projection import Grid, Lens
from .sky import LABELS, SKY_COLORS


PALE = (241, 239, 224)
MUTED = (158, 179, 177)
PANEL = (31, 43, 48)
DARK = (20, 31, 36)
MINT = (144, 224, 175)
GOLD = (254, 203, 125)
PINK = (246, 147, 152)


def _draw_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def tinted_rgb(image: np.ndarray, codes: np.ndarray, opacity: float = .47) -> np.ndarray:
    mapped = SKY_COLORS[codes]
    native = np.clip(image * 255, 0, 255)
    return np.rint(native * (1 - opacity) + mapped * opacity).astype(np.uint8)


def _image(data: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(data, dtype=np.uint8))


def _on_background(image: Image.Image, color: tuple[int, int, int] = PANEL) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGB")
    canvas = Image.new("RGB", image.size, color)
    canvas.paste(image, mask=image.getchannel("A"))
    return canvas


def photo_canvas(camera: Camera, edge: int = 920) -> Image.Image:
    return _on_background(camera.code_preview(edge))


def _native(points: np.ndarray, image: Image.Image) -> np.ndarray:
    return points * [image.width - 1, image.height - 1]


def plot_over_camera(camera: Camera, points: Iterable[tuple[float, float, str, tuple]], *,
                     edge: int = 920) -> Image.Image:
    image = photo_canvas(camera, edge)
    draw = ImageDraw.Draw(image)
    for u, v, name, tone in points:
        if not 0 <= u <= 1 or not 0 <= v <= 1:
            continue
        x, y = _native(np.array([u, v]), image)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=tone, outline=DARK, width=2)
        draw.text((x + 8, y - 9), name, fill=PALE, stroke_width=2, stroke_fill=DARK, font=_draw_font(13))
    return image


def calibration_view(camera: Camera, lens: Lens) -> Image.Image:
    image = photo_canvas(camera, 860)
    draw = ImageDraw.Draw(image)
    for elevation in (0, 15, 30, 45, 60, 75):
        bearings = np.arange(0, 362, 2)
        direction = np.stack((bearings, np.full(len(bearings), elevation)), axis=-1)
        coords, seen = lens.world_to_image(direction, camera)
        line_with_gaps(draw, _native(coords, image), seen, MINT if elevation == 0 else (121, 168, 155), 2)
    for degree, letter in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
        e = np.arange(0, 91, 1)
        coords, seen = lens.world_to_image(np.stack((np.full(len(e), degree), e), axis=-1), camera)
        line_with_gaps(draw, _native(coords, image), seen, GOLD, 2)
        first = np.flatnonzero(seen)
        if len(first):
            x, y = _native(coords[first[min(len(first) - 1, 6)]], image)
            draw.text((x + 4, y + 4), letter, fill=GOLD, stroke_width=2, stroke_fill=DARK, font=_draw_font(16))
    return image


def line_with_gaps(draw: ImageDraw.ImageDraw, points: np.ndarray, accepted: np.ndarray,
                   color: tuple[int, int, int], width: int = 2) -> None:
    last = None
    for point, allowed in zip(points, accepted, strict=True):
        if not allowed or not np.all(np.isfinite(point)):
            last = None
            continue
        current = tuple(point)
        if last is not None:
            draw.line([last, current], fill=color, width=width)
        last = current


def draw_marks(camera: Camera, lens: Lens, marks: list[dict]) -> Image.Image:
    points = []
    for mark in marks:
        note = mark["title"]
        tone = {"opening": MINT, "obstruction": PINK, "reference": GOLD, "note": PALE}[mark["role"]]
        if mark.get("native") is not None:
            u, v = mark["native"]
        else:
            projected, support = lens.world_to_image(np.array(mark["direction"]), camera)
            if not support:
                continue
            u, v = projected
        points.append((u, v, note[:25], tone))
    return plot_over_camera(camera, points)


def _bearing_ticks(canvas, region, origin, sweep):
    draw = ImageDraw.Draw(canvas)
    x, y, width, height = region
    draw.line((x, y + height, x + width, y + height), MINT, width=1)
    count = 4 if sweep < 160 else 8
    for i in range(count + 1):
        location = x + round(i * width / count)
        angle = (origin + sweep * i / count) % 360
        draw.line((location, y + height, location, y + height + 4), MUTED)
        draw.text((location, y + height + 6), f"{angle:g}°", font=_draw_font(13),
                  fill=MUTED, anchor="mt")


def _elevation_ticks(canvas, region, grid):
    draw = ImageDraw.Draw(canvas)
    x, y, _, height = region
    lo, hi = grid.footprint.bottom, grid.footprint.top
    low_s, high_s = grid.sine_edges[[0, -1]]
    for item in range(0, 91, 15):
        if not lo - .01 <= item <= hi + .01:
            continue
        weight = (np.sin(np.radians(item)) - low_s) / (high_s - low_s)
        v = y + height - round(height * weight)
        draw.line((x - 4, v, x, v), MUTED)
        draw.text((x - 7, v), f"{item}°", fill=MUTED, font=_draw_font(12), anchor="rm")


def angular_map(colors: np.ndarray, grid: Grid, title: str, legend: dict | None = None) -> Image.Image:
    chart = Image.new("RGB", (960, 490), DARK)
    x, y, width, height = 69, 57, 835, 352
    layer = _image(colors).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    layer = layer.resize((width, height), resample=Image.Resampling.NEAREST)
    chart.paste(layer, (x, y))
    draw = ImageDraw.Draw(chart)
    draw.text((x, 20), title, fill=PALE, font=_draw_font(23))
    draw.rectangle((x - 1, y - 1, x + width, y + height), outline=MUTED, width=1)
    _bearing_ticks(chart, (x, y, width, height), grid.footprint.bearing, grid.footprint.sweep)
    _elevation_ticks(chart, (x, y, width, height), grid)
    if legend:
        location = x
        for name, color in legend.items():
            draw.rectangle((location, 452, location + 16, 466), fill=tuple(color))
            draw.text((location + 21, 449), name, fill=PALE, font=_draw_font(14))
            location += int(draw.textlength(name, font=_draw_font(14))) + 62
    return chart


def category_map(codes: np.ndarray, grid: Grid, title: str) -> Image.Image:
    glossary = dict(zip(("No measured pixel", "Obstruction", "Open sky", "Uncertain"), SKY_COLORS, strict=True))
    return angular_map(SKY_COLORS[codes], grid, title, glossary)


def height_chart(series: list[dict], title: str, ylabel: str, xlabel: str,
                 *, reference: float | None = None, hard_bounds: tuple[float, float] | None = None) -> Image.Image:
    width, height = 920, 430
    chart = Image.new("RGB", (width, height), DARK)
    draw = ImageDraw.Draw(chart)
    x, y = 72, 67
    plot_width, plot_height = 788, 260
    values = [float(v) for entry in series for v in entry["values"] if v is not None and np.isfinite(v)]
    if hard_bounds:
        lower, upper = hard_bounds
    elif values:
        lower, upper = min(values), max(values)
        margin = max((upper - lower) * .10, .004)
        lower, upper = lower - margin, upper + margin
    else:
        lower, upper = 0., 1.
    draw.rectangle((x, y, x + plot_width, y + plot_height), PANEL)
    draw.text((x, 24), title, fill=PALE, font=_draw_font(24))
    draw.text((x + plot_width, y + plot_height + 26), xlabel, fill=MUTED, anchor="rt", font=_draw_font(13))
    draw.text((x, y - 5), ylabel, fill=MUTED, anchor="lb", font=_draw_font(13))
    for i in range(6):
        location = y + round(plot_height * i / 5)
        draw.line((x, location, x + plot_width, location), (53, 68, 72))
        amount = upper - (upper - lower) * i / 5
        draw.text((x - 5, location), f"{amount:.3g}", font=_draw_font(13), fill=MUTED, anchor="rm")
    if reference is not None and lower <= reference <= upper:
        line = y + plot_height * (upper - reference) / (upper - lower)
        draw.line((x, line, x + plot_width, line), MUTED)
    count = max((len(entry["values"]) for entry in series), default=1)
    xs = x + (np.arange(count) * plot_width / max(1, count - 1))
    location = x
    for entry in series:
        arr = np.asarray([np.nan if v is None else v for v in entry["values"]])
        ys = y + (upper - arr) * plot_height / (upper - lower)
        line_with_gaps(draw, np.stack((xs[:len(arr)], ys), axis=-1), np.isfinite(arr),
                       entry["color"], 3)
        draw.line((location, 389, location + 26, 389), entry["color"], width=3)
        draw.text((location + 35, 381), entry["name"], font=_draw_font(14), fill=PALE)
        location += int(draw.textlength(entry["name"], font=_draw_font(14))) + 85
    return chart


def comparison_panel(panels: list[tuple[str, Image.Image]], title: str) -> Image.Image:
    width = 980
    per = (width - 32 - 12 * (len(panels) - 1)) // len(panels)
    converted = []
    for text, source in panels:
        image = _on_background(source)
        ratio = per / image.width
        resized = image.resize((per, max(1, round(image.height * ratio))), Image.Resampling.LANCZOS)
        converted.append((text, resized))
    area = max(image.height for _, image in converted)
    panel = Image.new("RGB", (width, area + 102), DARK)
    draw = ImageDraw.Draw(panel)
    draw.text((16, 16), title, fill=PALE, font=_draw_font(23))
    for i, (label, image) in enumerate(converted):
        left = 16 + i * (per + 12)
        draw.text((left, 58), label[:40], fill=PALE, font=_draw_font(16))
        panel.paste(image, (left, 86 + (area - image.height) // 2))
    return panel


def sun_path_on_photo(camera: Camera, locations: np.ndarray, status: np.ndarray, measured: np.ndarray) -> Image.Image:
    image = photo_canvas(camera)
    draw = ImageDraw.Draw(image)
    positions = _native(locations, image)
    support = np.isfinite(positions).all(axis=-1) & ((locations >= 0) & (locations <= 1)).all(axis=-1)
    line_with_gaps(draw, positions, support & measured, PALE, 2)
    stride = max(1, len(positions) // 130)
    for index in range(0, len(positions), stride):
        if not support[index]:
            continue
        u, v = positions[index]
        tone = tuple(SKY_COLORS[status[index]])
        draw.ellipse((u - 3.5, v - 3.5, u + 3.5, v + 3.5), fill=tone, outline=DARK, width=1)
    return image


def normalized_heat(values: np.ndarray, seen: np.ndarray, low: float = 0., high: float = 1.) -> np.ndarray:
    setting = np.where(seen, (values - low) / max(1e-5, high - low), 0)
    intensity = np.clip(setting, 0, 1)
    blue = np.array([49, 78, 116], dtype=float)
    gold = np.array([253, 213, 119], dtype=float)
    shade = blue + intensity[..., None] * (gold - blue)
    return np.where(seen[..., None], shade, Palette().missing).astype(np.uint8)


def toned_raw(rgb: np.ndarray, seen: np.ndarray) -> np.ndarray:
    visually = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return np.where(seen[..., None], visually, np.asarray(Palette().missing, dtype=np.uint8))


def border_color_at(index: int) -> tuple[int, int, int]:
    angle = (index * .61803398875 % 1) * 2 * math.pi
    r = round(163 + math.cos(angle) * 57)
    g = round(174 + math.cos(angle + 2 * math.pi / 3) * 57)
    b = round(172 + math.cos(angle + 4 * math.pi / 3) * 57)
    return r, g, b
