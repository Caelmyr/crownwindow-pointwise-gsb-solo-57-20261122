"""Deterministic overhead photographs for the local training workbench."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .camera import Camera, encode_png
from .contracts import Capture, Station
from .library import Library
from .projection import Lens


@dataclass(frozen=True)
class Demo:
    key: str
    title: str
    description: str
    kind: str
    seed: int

    def public(self) -> dict:
        return {"key": self.key, "title": self.title, "description": self.description,
                "preview": f"/api/examples/{self.key}/preview", "kind": self.kind, "synthetic": True}


EXAMPLES = {
    example.key: example for example in (
        Demo("ash-lane", "Ash Lane canopy", "Sunlit foliage, branch gaps and a clipped boundary", "woodland", 2704),
        Demo("courtyard", "Brick courtyard", "Directional openings and a new temporary screen", "courtyard", 9063),
        Demo("field-edge", "Field edge", "Low branches and complementary missing parts", "border", 7001),
        Demo("paired-cover", "Strap and clip pair", "Same station twice: a strap blocks the left side, a clip blocks the right, and wind moves one leaf patch", "merge", 5117),
    )
}


@dataclass(frozen=True)
class Scene:
    capture: Capture
    lens: Lens
    file: str
    columns: int
    rows: int
    changed: bool


def demo_frames(example: Demo) -> tuple[Scene, ...]:
    gallery = (
        Scene(Capture("First visit · north mark", "2026-06-09T07:30:00+00:00", "Level overhead camera", "survey", "First visit"),
              Lens((.498, .508), .477, 90., 10., "equidistant"), "visit-one.png", 760, 760, False),
        Scene(Capture("Second visit · rotated lens", "2026-06-11T12:35:00+00:00", "Second view of the same point", "comparison", "Second visit"),
              Lens((.505, .495), .55, 86., -24 % 360, "equisolid"), "visit-two.png", 820, 710, True),
    )
    if example.key == "paired-cover":
        gallery = (
            Scene(Capture("First frame · strap on the left", "2026-07-02T08:10:00+00:00", "Same tripod station, left side blocked by the strap", "survey", "First frame"),
                  Lens((.5, .5), .48, 90., 0., "equidistant"), "pair-one.png", 780, 780, False),
            Scene(Capture("Second frame · clip on the right", "2026-07-02T08:11:00+00:00", "Same tripod station, right side blocked by the clip", "comparison", "Second frame"),
                  Lens((.5, .5), .48, 90., 0., "equidistant"), "pair-two.png", 780, 780, True),
        )
    return gallery


def _ring_delta(bearing, center):
    return (bearing - center + 180) % 360 - 180


def _make_sky(bearing, elevation, update: bool, rng, kind: str = ""):
    radiant = np.sin(np.radians(elevation.clip(0, 90)))
    halo = np.exp(-(_ring_delta(bearing, 163) / 30) ** 2 - ((elevation - 46) / 26) ** 2)
    blue = np.stack((.62 + .10 * (1 - radiant), .77 + .055 * (1 - radiant), .89 + .035 * (1 - radiant)), axis=-1)
    cloud = .07 * (.3 + .7 * np.sin(np.radians(bearing * 4.7 + elevation * 5)) ** 2)
    cloud *= np.sin(np.radians(elevation * 2 + bearing / 3)) ** 2
    cloud += .055 * np.sin(np.radians(bearing * 1.4 - elevation * 15)) ** 12
    blue += cloud[..., None] * np.array([1, .79, .48])
    blue += halo[..., None] * [.075, .07, .035]
    if update and kind != "merge":
        blue *= [.935, .958, .964]
    blue += rng.normal(0, .004, elevation.shape)[..., None]
    return blue


def _simulate_canopy(kind: str, seed: int, update: bool) -> tuple[np.ndarray, np.ndarray]:
    width, height = 1536, 420
    bear, alt = np.meshgrid(np.linspace(0, 360, width, endpoint=False), np.linspace(0, 90, height))
    env = np.random.default_rng(seed)
    if kind == "courtyard":
        sky_line = 19 + np.where((_ring_delta(bear, 290) < 16) & (_ring_delta(bear, 290) > -42), 23, 0)
        sky_line += np.where((_ring_delta(bear, 115) < 37) & (_ring_delta(bear, 115) > -18), 12, 0)
    else:
        sky_line = 13 + 8 * np.sin(np.radians(bear * 2.8)) ** 2 + 6 * np.sin(np.radians(bear * 4.1 + 17)) ** 2
    solid = alt < sky_line
    palette = np.zeros((*bear.shape, 3), dtype=np.float32)
    temper = .5 + .5 * np.sin(np.radians(alt * 8 + bear * 4))
    low = np.stack((.08 + .10 * temper, .15 + .15 * temper, .10 + .065 * temper), axis=-1)
    palette[solid] = low[solid]
    draw_base = Image.fromarray(np.uint8(np.clip(palette * 255, 0, 255)))
    shadows = Image.fromarray(solid.astype(np.uint8) * 255)
    leaves = ImageDraw.Draw(draw_base)
    flags = ImageDraw.Draw(shadows)
    clusters = [(34, 40), (76, 35), (119, 49), (182, 40), (231, 55), (266, 34), (314, 48), (354, 38)]
    if kind == "border":
        clusters = [(21, 42), (49, 47), (108, 35), (249, 32), (297, 53), (337, 62)]
    if kind == "merge":
        clusters = [(28, 44), (74, 52), (126, 40), (178, 58), (227, 46), (276, 55), (326, 41), (352, 63)]
    if kind == "courtyard":
        clusters = [(21, 41), (59, 66), (320, 51), (351, 52)]
        _screen(leaves, flags, width, height, update)
    if kind == "merge" and not update:
        # A compact leaf cluster that the wind moves out of frame two's evidence.
        bx, by = 185 * width / 360, 52 * (height - 1) / 90
        leaves.ellipse((bx - 40, by - 30, bx + 40, by + 30), fill=(41, 88, 47))
        flags.ellipse((bx - 40, by - 30, bx + 40, by + 30), fill=255)
    for i, (b, h) in enumerate(clusters):
        if update and i == 1 and kind != "merge":
            _branch(leaves, flags, width, height, b + 6, h - 4, seed + i)
        else:
            _branch(leaves, flags, width, height, b, h, seed + i)
        count = 170 if kind != "courtyard" else 130
        for _ in range(count):
            bearing = env.normal(b, 16 + i % 3)
            elev = env.normal(h, 10)
            horizontal = env.uniform(.4, 3.3)
            vertical = env.uniform(.45, 2.9)
            tint = max(0., np.sin(elev / 22 + bearing / 27) + env.uniform(-.5, .55))
            color = (round(23 + 37 * tint + env.integers(0, 36)),
                     round(44 + 54 * tint + env.integers(0, 49)),
                     round(22 + 20 * tint + env.integers(0, 30)))
            if update and kind != "merge" and 0 < i < 3 and 55 < bearing < 91 and elev > 36:
                continue
            if kind == "merge" and update and 169 < bearing < 201 and 41 < elev < 63:
                # The wind-moved patch: leaves stay in frame one, sky in frame two.
                continue
            for shift in (-360, 0, 360):
                x = (bearing + shift) * width / 360
                y = elev * (height - 1) / 90
                if x < -20 or x > width + 20 or y < 0 or y > height:
                    continue
                rectangle = (x - horizontal * 2.4, y - vertical * 2.4, x + horizontal * 2.4, y + vertical * 2.4)
                leaves.ellipse(rectangle, fill=color)
                flags.ellipse(rectangle, fill=255)
    return np.asarray(draw_base).astype(np.float32) / 255, np.asarray(shadows).astype(bool)


def _branch(leaves, flags, width: int, height: int, bearing: float, top: float, seed: int) -> None:
    random = np.random.default_rng(seed)
    shape = [(bearing + 26, 4), (bearing + 14, top * .53), (bearing, top),
             (bearing - 12, top + 15)]
    for shift in (-360, 0, 360):
        section = [((b + shift) * width / 360, a * (height - 1) / 90) for b, a in shape]
        for i, (a, b) in enumerate(zip(section[:-1], section[1:], strict=True)):
            line = max(2, 22 - i * 7)
            leaves.line((a, b), fill=(51 + i * 6, 57 + i * 5, 37), width=line)
            flags.line((a, b), fill=255, width=line)
        for branch in range(8):
            start = shape[1] if branch < 4 else shape[2]
            b = start[0] + random.uniform(-27, 24)
            el = start[1] + random.uniform(5, 24)
            sx = (start[0] + shift) * width / 360
            sy = start[1] * (height - 1) / 90
            ex = (b + shift) * width / 360
            ey = el * (height - 1) / 90
            leaves.line((sx, sy, ex, ey), fill=(62, 70, 39), width=3)
            flags.line((sx, sy, ex, ey), fill=255, width=3)


def _screen(leaves, mask, width: int, height: int, update: bool):
    rectangle = ((135 if update else 143) * width / 360, 0, 168 * width / 360, (48 if update else 31) * height / 90)
    leaves.rectangle(rectangle, fill=(94, 63, 44))
    mask.rectangle(rectangle, fill=255)


def _scene_direction(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    uv_x, uv_y = np.meshgrid(np.linspace(0, 1, scene.columns), np.linspace(0, 1, scene.rows))
    positions = np.stack((uv_x, uv_y), axis=-1)
    dummy = Camera(np.empty(0), np.empty(0), scene.columns, scene.rows, "image/png", "", "", "PNG", "sRGB", 0)
    return scene.lens.image_to_world(positions, dummy)


def _opaque_mask(example: Demo, scene: Scene, direction: np.ndarray, geometric: np.ndarray):
    bearing, elevation = np.moveaxis(direction, -1, 0)
    valid = geometric.copy()
    if example.kind == "merge":
        # Same station, same lens: the two frames only differ by the accessory
        # blocking one side and by one wind-moved leaf patch in the overlap.
        if not scene.changed:
            # Camera strap hanging into the left (western) side of frame one.
            valid &= ~((_ring_delta(bearing, 252) > -26) & (_ring_delta(bearing, 252) < 22)
                       & (elevation < 52))
        else:
            # A clip mounted on the right (eastern) side of frame two.
            valid &= ~((_ring_delta(bearing, 72) > -20) & (_ring_delta(bearing, 72) < 24)
                       & (elevation < 55))
            # The wind patch stays visible: its leaves simply are not drawn in
            # frame two, so the camera records open sky at those directions.
        return valid
    if scene.changed:
        valid &= ~((_ring_delta(bearing, 248) > -14) & (_ring_delta(bearing, 248) < 23) & (elevation < 36))
        valid &= ~(np.hypot(_ring_delta(bearing, 312) / 28, (elevation - 37) / 12) < 1)
    else:
        valid &= ~(np.hypot(_ring_delta(bearing, 82) / 22, (elevation - 29) / 16) < 1)
    if example.kind == "border":
        valid &= ~((_ring_delta(bearing, 154) > -19) & (_ring_delta(bearing, 154) < 22) & (elevation < 27))
    return valid


@lru_cache(maxsize=8)
def render_observation(sample: str, changed: bool, thumbnail: bool = False) -> bytes:
    example = EXAMPLES[sample]
    scene = demo_frames(example)[int(changed)]
    if thumbnail:
        scene = Scene(scene.capture, scene.lens, scene.file, 400, 400, changed)
    direction, geometric = _scene_direction(scene)
    az, el = np.moveaxis(direction, -1, 0)
    canopy, marked = _simulate_canopy(example.kind, example.seed, changed)
    az_idx = (np.rint(az / 360 * canopy.shape[1]).astype(int)) % canopy.shape[1]
    el_idx = np.clip(np.rint(el / 90 * (canopy.shape[0] - 1)).astype(int), 0, canopy.shape[0] - 1)
    random = np.random.default_rng(example.seed + int(changed))
    sky = _make_sky(az, el, changed, random, example.kind)
    terrain = np.where(marked[el_idx, az_idx, None], canopy[el_idx, az_idx], sky)
    noise = np.array([random.normal(0, .009, el.shape)] * 3).transpose((1, 2, 0))
    image = np.clip((terrain + noise) * 255, 0, 255).astype(np.uint8)
    for i in range(3):
        image[..., i] = np.rint(ndimage.gaussian_filter(image[..., i].astype(float), .60)).astype(np.uint8)
    coverage = _opaque_mask(example, scene, direction, geometric)
    return encode_png(np.dstack((image, coverage.astype(np.uint8) * 255)))


def populate(library: Library, key: str) -> tuple[dict, bool]:
    example = EXAMPLES[key]
    station, created = library.create_station(Station(example.title, "TRAINING-" + str(example.seed),
                                                     example.description, True), demo_key=key)
    details = library.station_details(station["id"])
    by_title = {photo["capture"]["title"]: photo for photo in details["photos"]}
    # A previously interrupted import reuses cameras that are already in place.
    for scene in demo_frames(example):
        saved = by_title.get(scene.capture.title)
        if saved is None:
            raw = render_observation(key, scene.changed)
            saved = library.add_photo(station["id"], Camera.read(raw, scene.file), raw, scene.capture)
        if not saved["active_lens"]:
            library.add_lens(saved["id"], saved["revision"], "Synthetic field geometry", scene.lens,
                             {"method": "known synthetic geometry", "sample": key})
    return library.station_details(station["id"])["station"], created
