"""Native pixels, coverage, transfer functions and safe camera decoding."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from scipy import ndimage

from .contracts import Line, Problem, filename, number, sha


MAX_FILE_BYTES = 24 * 1024 * 1024
MAX_PIXELS = 12_000_000
MIN_SIDE = 24
FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "TIFF": "image/tiff",
    "WEBP": "image/webp",
}
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, np.float32)
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def rgb_luminance(rgb: np.ndarray) -> np.ndarray:
    return np.sum(rgb * np.array([.2126, .7152, .0722], np.float32), axis=-1)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.maximum(np.asarray(rgb, np.float32), 0)
    return np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - .055)


def encode_png(data: np.ndarray | Image.Image) -> bytes:
    out = io.BytesIO()
    image = data if isinstance(data, Image.Image) else Image.fromarray(data)
    image.save(out, format="PNG")
    return out.getvalue()


def bounded_size(width: int, height: int, edge: int) -> tuple[int, int]:
    if max(width, height) <= edge:
        return width, height
    factor = edge / max(width, height)
    return max(1, round(width * factor)), max(1, round(height * factor))


def _standard_color(source: Image.Image) -> Image.Image:
    rgb = source.convert("RGB")
    embedded = source.info.get("icc_profile")
    if not embedded:
        return rgb
    try:
        target = ImageCms.createProfile("sRGB")
        profile = ImageCms.ImageCmsProfile(io.BytesIO(embedded))
        base = source if source.mode in ("RGB", "CMYK", "L") else rgb
        return ImageCms.profileToProfile(base, profile, target, outputMode="RGB")
    except (ImageCms.PyCMSError, OSError, ValueError, TypeError) as error:
        raise Problem("The embedded color profile could not be converted to sRGB.") from error


def _sixteen_bit_image(image: Image.Image) -> np.ndarray | None:
    if image.mode not in ("I", "I;16", "I;16B", "I;16L"):
        return None
    scalar = np.asarray(image)
    if scalar.min() < 0 or scalar.max() > 65535:
        raise Problem("Only unsigned 16-bit grayscale camera values are supported.")
    return np.repeat((scalar.astype(np.float32) / 65535)[..., None], 3, axis=-1)


@dataclass(frozen=True)
class Camera:
    """Decoded, oriented pixels and separate measured coverage."""

    color: np.ndarray
    valid: np.ndarray
    width: int
    height: int
    mime: str
    original_name: str
    content_sha: str
    format: str
    encoding: str
    invalid_share: float

    @classmethod
    def read(cls, raw: bytes, original_name: str) -> Camera:
        if not raw or len(raw) > MAX_FILE_BYTES:
            raise Problem("An image file must be non-empty and no larger than 24 MiB.", 413)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as image:
                    return cls.decode(image, original_name, sha(raw))
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            raise Problem("The camera image exceeds the 12 megapixel budget.", 413) from error
        except (UnidentifiedImageError, OSError, SyntaxError) as error:
            raise Problem("This camera image could not be decoded.") from error

    @classmethod
    def decode(cls, image: Image.Image, original_name: str, digest: str) -> Camera:
        format_name = image.format
        mime = FORMATS.get(format_name)
        if not mime:
            raise Problem("Choose PNG, JPEG, WebP, or single-frame TIFF.")
        if getattr(image, "n_frames", 1) != 1:
            raise Problem("Import individual images instead of a multi-frame image.")
        width, height = image.size
        if min(width, height) < MIN_SIDE:
            raise Problem("A camera image needs at least 24 pixels on each side.")
        if width * height > MAX_PIXELS:
            raise Problem("The camera image exceeds the 12 megapixel budget.", 413)
        source = ImageOps.exif_transpose(image)
        width, height = source.size
        if source.mode in ("RGBA", "LA", "PA") or "transparency" in source.info:
            alpha = np.asarray(source.convert("RGBA").getchannel("A"))
            valid = alpha >= 250
        else:
            valid = np.ones((height, width), dtype=bool)
        if not np.any(valid):
            raise Problem("The image contains no opaque measured pixels.")
        rgb = _sixteen_bit_image(source)
        if rgb is None:
            rgb = np.asarray(_standard_color(source), dtype=np.float32) / 255
            encoding = "sRGB"
        else:
            encoding = "grayscale code values"
        return cls(rgb, valid, width, height, mime,
                   filename(original_name, "capture"), digest, format_name, encoding,
                   float(1 - np.mean(valid)))

    @property
    def scale(self) -> np.ndarray:
        return np.array([self.width - 1, self.height - 1])

    def pixel_to_unit(self, coordinates: np.ndarray) -> np.ndarray:
        return np.asarray(coordinates, dtype=float) / self.scale

    def unit_to_pixel(self, coordinates: np.ndarray) -> np.ndarray:
        return np.asarray(coordinates, dtype=float) * self.scale

    def public(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "mime": self.mime,
            "file": self.original_name,
            "sha256": self.content_sha,
            "format": self.format,
            "encoding": self.encoding,
            "invalid_share": number(self.invalid_share),
        }

    def code_preview(self, edge: int = 1000) -> Image.Image:
        color = np.rint(np.clip(self.color, 0, 1) * 255).astype(np.uint8)
        alpha = self.valid.astype(np.uint8) * 255
        image = Image.fromarray(np.dstack((color, alpha)))
        return image.resize(bounded_size(self.width, self.height, edge), Image.Resampling.LANCZOS)

    def read_nearest(self, uv: tuple[float, float]) -> dict:
        x, y = self.unit_to_pixel(uv)
        ix, iy = int(round(x)), int(round(y))
        valid = bool(self.valid[iy, ix])
        if not valid:
            return {"unit": list(uv), "pixel": [ix, iy], "valid": False, "rgb": None}
        rgb = self.color[iy, ix]
        return {
            "unit": list(uv),
            "pixel": [ix, iy],
            "valid": True,
            "rgb": [number(v, 6) for v in rgb],
            "display_luminance": number(rgb_luminance(srgb_to_linear(rgb))),
        }

    def support_window(self, center: tuple[float, float], pixels: float) -> dict:
        x, y = self.unit_to_pixel(center)
        radius = max(1, round(pixels))
        x0, x1 = max(0, round(x) - radius), min(self.width, round(x) + radius + 1)
        y0, y1 = max(0, round(y) - radius), min(self.height, round(y) + radius + 1)
        in_bounds = self.valid[y0:y1, x0:x1]
        color = self.color[y0:y1, x0:x1][in_bounds]
        if color.size:
            low, middle, high = np.quantile(color, [.1, .5, .9], axis=0)
            median = [number(v, 6) for v in middle]
            spread = [number(v, 6) for v in high - low]
        else:
            median = spread = None
        return {
            "bounds_px": [x0, y0, x1, y1],
            "expected_pixels": (2 * radius + 1) ** 2,
            "supported_pixels": int(in_bounds.sum()),
            "median_rgb": median,
            "central_eighty_percent_width": spread,
        }

    def bilinear(self, uv: np.ndarray, *, minimum: float = .995) -> tuple[np.ndarray, np.ndarray]:
        """Sample only visible support. Transparent neighbours do not darken a valid pixel."""
        return self.bilinear_pixels(self.unit_to_pixel(uv), minimum=minimum)

    def bilinear_pixels(self, xy: np.ndarray, *, minimum: float = .995) -> tuple[np.ndarray, np.ndarray]:
        xy = np.asarray(xy, dtype=float)
        x, y = xy[..., 0], xy[..., 1]
        domain = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= self.width - 1) & (y >= 0) & (y <= self.height - 1)
        map_x = np.where(domain, x, -4)
        map_y = np.where(domain, y, -4)
        coordinates = np.stack((map_y, map_x))
        weight = ndimage.map_coordinates(self.valid.astype(np.float32), coordinates, order=1,
                                         mode="constant", cval=0, prefilter=False)
        supported = domain & (weight >= minimum)
        values = []
        for channel in range(self.color.shape[-1]):
            premultiplied = np.where(self.valid, self.color[..., channel], 0)
            readout = ndimage.map_coordinates(premultiplied, coordinates, order=1,
                                              mode="constant", cval=0, prefilter=False)
            values.append(np.where(supported, readout / np.maximum(weight, 1e-6), 0))
        return np.stack(values, axis=-1), supported

    def line_readout(self, line: Line, count: int = 121, width_px: float = 0) -> list[dict]:
        p, q = self.unit_to_pixel(line.public())
        delta = q - p
        length = np.linalg.norm(delta)
        perpendicular = np.array([-delta[1], delta[0]]) / max(length, 1e-9)
        steps = np.linspace(0, 1, count)
        core = p[None, :] + steps[:, None] * delta
        offsets = np.linspace(-width_px / 2, width_px / 2, 5) if width_px else np.array([0.])
        samples = core[:, None, :] + offsets[None, :, None] * perpendicular
        values, seen = self.bilinear_pixels(samples)
        luminance = rgb_luminance(srgb_to_linear(values))
        count_seen = seen.sum(axis=1)
        summed = np.where(seen, luminance, 0).sum(axis=1)
        rows = []
        for index, position in enumerate(core):
            visible = int(count_seen[index])
            rows.append({
                "index": index + 1,
                "distance_px": number(length * steps[index]),
                "pixel": [number(v, 3) for v in position],
                "unit": [number(v, 6) for v in self.pixel_to_unit(position)],
                "sample_count": visible,
                "expected": len(offsets),
                "luminance": number(summed[index] / visible) if visible else None,
            })
        return rows


def safe_download_label(camera: Camera) -> str:
    label = Path(camera.original_name).stem
    suffix = {"image/jpeg": ".jpg", "image/webp": ".webp", "image/tiff": ".tiff"}.get(camera.mime, ".png")
    return label + suffix


def display_scale(camera: Camera, floor: float, ceiling: float, gamma: float,
                  *, monochrome: bool = False) -> np.ndarray:
    values = np.clip((camera.color - floor) / (ceiling - floor), 0, 1) ** (1 / gamma)
    if monochrome:
        values = np.broadcast_to(rgb_luminance(values)[..., None], values.shape)
    rgb = np.rint(values * 255).astype(np.uint8)
    return np.dstack((rgb, camera.valid.astype(np.uint8) * 255))


def describe_rgb(values: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    percentiles = np.quantile(values, [0, .1, .5, .9, 1], axis=0)
    return {
        "count": int(len(values)),
        "minimum": [number(x, 5) for x in percentiles[0]],
        "p10": [number(x, 5) for x in percentiles[1]],
        "median": [number(x, 5) for x in percentiles[2]],
        "p90": [number(x, 5) for x in percentiles[3]],
        "maximum": [number(x, 5) for x in percentiles[4]],
    }
