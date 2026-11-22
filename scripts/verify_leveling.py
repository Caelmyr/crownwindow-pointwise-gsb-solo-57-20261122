#!/usr/bin/env python3
"""Standalone check for tilted-camera leveling and agreement holdouts.

Generates a synthetic level fisheye scene, applies a known optical-axis tilt to
make the "this time" capture, then recovers the tilt from a handful of surveyed
corners under both placements and verifies the agreement/holdout machinery.

Run from the repository root:  python3 scripts/verify_leveling.py
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image

from crownwindow.calibration import Landmark
from crownwindow.camera import Camera
from crownwindow.lean import lean_fit, lean_marks, lens_with_holdout, leveling_agreement
from crownwindow.projection import Lens, RasterHoldout
from crownwindow.sky import Classifier, Controls, sample_directions


SIZE = 720
TRUE_TILT = 5.5
TRUE_TILT_BEARING = 118.0
TRUE_NORTH = 24.0
THRESHOLD = 2.0


def make_camera() -> Camera:
    rng = np.random.default_rng(7)
    return Camera(rng.random((SIZE, SIZE, 3), np.float32),
                  np.ones((SIZE, SIZE), bool), SIZE, SIZE, "image/png",
                  "tilted-check.png", "synthetic", "PNG", "sRGB", 0.0)


def main() -> int:
    camera = make_camera()
    true_lens = Lens((.5, .5), .48, 90, TRUE_NORTH, "equidistant",
                     (TRUE_TILT, TRUE_TILT_BEARING), None)
    prior = Lens((.502, .497), .482, 90, 0.0, "equidistant")

    corners = [
        ("North wall", 12, 64), ("East corner", 96, 48),
        ("Southeast edge", 142, 30), ("Northwest corner", 308, 55),
    ]
    marks = []
    for index, (title, bearing, elevation) in enumerate(corners):
        uv, seen = true_lens.world_to_image(np.array([[bearing, elevation]], float), camera)
        assert bool(seen[0]), f"{title} landed outside the lens"
        marks.append({"title": title, "image": [float(uv[0, 0]), float(uv[0, 1])],
                      "direction": [bearing, elevation], "confidence": 1.0})
    landmarks = lean_marks(marks)

    fixed, fixed_fit, fw = lean_fit(camera, prior, landmarks, "fixed_circle")
    adjusted, adjusted_fit, aw = lean_fit(camera, prior, landmarks, "adjusted_circle")
    print("fixed    lean:", fixed.lean, "north:", round(fixed.north_clockwise, 3),
          "rms_px:", fixed_fit.metrics["rms_px"])
    print("adjusted lean:", adjusted.lean, "north:", round(adjusted.north_clockwise, 3),
          "rms_px:", adjusted_fit.metrics["rms_px"])
    tilt_error = abs(adjusted.lean[0] - TRUE_TILT)
    assert tilt_error < .15, f"adjusted tilt recovery error too large: {tilt_error}"
    bearing_error = abs((adjusted.lean[1] - TRUE_TILT_BEARING + 180) % 360 - 180)
    assert bearing_error < 1.5, f"adjusted tilt bearing error too large: {bearing_error}"
    north_error = abs((adjusted.north_clockwise - TRUE_NORTH + 180) % 360 - 180)
    assert north_error < .5, f"north recovery error too large: {north_error}"
    # The fixed circle keeps the biased prior circle and absorbs the mismatch
    # into its tilt; the agreement machinery bounds exactly that divergence.
    assert abs(fixed.lean[0] - TRUE_TILT) < .5
    assert float(adjusted_fit.metrics["rms_px"]) < 1.0

    analysis = leveling_agreement(camera, fixed, adjusted, fixed_fit, adjusted_fit, fw, aw, THRESHOLD)
    print("summary:", analysis["summary"])
    for name, image in analysis["images"].items():
        Image.fromarray(np.asarray(image)).save(f"/tmp/leveling-{name}.png")
    assert analysis["summary"]["concordant_percent"] > 40

    raster = RasterHoldout.parse(analysis["holdouts"]["cautious"])
    withheld_lens = lens_with_holdout(fixed, raster)
    directions = analysis_grid()
    samples = sample_directions(camera, withheld_lens,
                                Classifier.train(camera, withheld_lens, Controls.parse(None)), directions)
    expected = raster.withheld(directions)
    missing = samples.codes == 0
    assert np.all(missing[expected]), "every holdout cell must be withheld as missing support"
    assert int(expected.sum()) > 0
    plain = sample_directions(camera, fixed, Classifier.train(camera, fixed, Controls.parse(None)), directions)
    assert int((plain.codes == 0).sum()) < int(missing.sum())

    one_side = tuple(Landmark.parse(mark, i) for i, mark in enumerate(marks[:3]))
    _, _, side_warnings = lean_fit(camera, prior, one_side, "fixed_circle")
    assert side_warnings, "one-sided corners must warn that the disputed belt stays withheld"

    payload = withheld_lens.public()
    again = Lens.parse(payload)
    assert again.public() == payload
    print("holdout cells withheld as MISSING:", int(expected.sum()), "of", expected.size)
    print("OK: tilt recovered, agreement mapped, holdout honored, lens round-trips.")
    return 0


def analysis_grid():
    bearings = (np.arange(180) + .5) * 2
    elevations = (np.arange(45) + .5) * 2
    b, e = np.meshgrid(bearings, elevations)
    return np.stack((b, e), axis=-1)


if __name__ == "__main__":
    sys.exit(main())
