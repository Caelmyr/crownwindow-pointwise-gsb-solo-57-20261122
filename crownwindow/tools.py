"""Tool catalog for the observation bench and its ordinary browser controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .apertures import windows
from .bench import Bench
from .concordance import DEMO_TIES, compare_ties
from .coverage import angular_share, classify_photo, horizon_line, inspect_geometry, inspect_source, line_profile
from .field import field_overlay
from .lighting import diffuse_reference, refind_change
from .optics import DEMO_BRIGHTNESS, DEMO_OPTIC_MARKS, lens_model_audit, relative_light_audit
from .patches import native_window
from .planes import DEMO_PLANES, oriented_surfaces
from .paths import DEMO_PATH, direct_path, transect
from .registration import DEMO_CORNERS, wall_registration
from .result import Result
from .sharpness import check_samples, field_detail
from .viewpoint import virtual_view


@dataclass(frozen=True)
class Field:
    key: str
    title: str
    kind: str
    default: object
    hint: str
    step: float | None = None

    def public(self) -> dict:
        return self.__dict__


def number(key, title, default, step, hint="") -> Field:
    return Field(key, title, "number", default, hint, step)


def toggle(key, title, default, hint="") -> Field:
    return Field(key, title, "checkbox", default, hint)


def mark(key, title, default, hint="") -> Field:
    return Field(key, title, "line", default, hint)


def structured(key, title, default, hint="", shape="structured") -> Field:
    return Field(key, title, shape, default, hint)


@dataclass(frozen=True)
class Tool:
    id: str
    label: str
    group: str
    photos: int
    settings: tuple[Field, ...]
    explain: str
    run: Callable[[Bench], Result]
    footprint: bool = False

    def public(self) -> dict:
        return {"id": self.id, "label": self.label, "group": self.group, "photos": self.photos,
                "settings": [field.public() for field in self.settings],
                "explain": self.explain, "footprint": self.footprint}


TOOLS = {value.id: value for value in (
    Tool("source-check", "Source signal", "Camera", 1, (
        number("floor", "Display floor", 0, .005),
        number("ceiling", "Display ceiling", 1, .005),
        number("gamma", "Display curve", 1, .05),
        toggle("monochrome", "Monochrome display", False),
        number("threshold", "Clipping level", .985, .005),
    ), "Read RGB code statistics, clipping and missing pixels. Adjust only the display window.", inspect_source),
    Tool("line-profile", "Source line profile", "Camera", 1, (
        mark("line", "Line endpoints", [[.16, .42], [.84, .42]], "Mark two points on the photograph."),
        number("samples", "Readout positions", 121, 1),
        number("width", "Mean band width (px)", 0, .25),
    ), "Read native pixel values along a marked line or narrow band. Archived chart points have exact support.", line_profile),
    Tool("lens-support", "Lens coverage", "Camera", 1, (),
         "Compare the field of view, transparent holes and opaque pixels, with the lens drawn on the photo.", inspect_geometry, True),
    Tool("sky-labels", "Sky evidence classes", "Canopy", 1, (),
         "Review where the camera supports a sky or obstruction decision. Set color samples in Prepare.", classify_photo),
    Tool("open-share", "Open sky area", "Canopy", 1, (
        number("sectors", "Bearing sectors", 8, 1),
        number("bands", "Elevation bands", 4, 1),
    ), "Measure solid-angle sky share and report bounds for uncertain coverage.", angular_share, True),
    Tool("horizon", "Obstruction skyline", "Canopy", 1, (
        number("bearings", "Bearing samples", 144, 1),
        number("elevations", "Height samples", 91, 1),
    ), "Find the highest camera-supported obstruction in each bearing and the next decisive sky.", horizon_line, True),
    Tool("sky-windows", "Canopy openings", "Canopy", 1, (
        number("minimum", "Minimum solid angle (sr)", .002, .0005),
        toggle("diagonal", "Connect touching corners", False),
    ), "Group measured sky into connected openings, with optional corner connectivity.", windows, True),
    Tool("sun-path", "Direct light path", "Light", 1, (
        structured("landmarks", "Timed direction marks", DEMO_PATH, "Use ISO timestamps with an explicit time zone and bearing/elevation pairs."),
        number("step", "Maximum probe interval (seconds)", 240, 60),
    ), "Check an entered direction path against opaque photo pixels and show resolved time bounds.", direct_path),
    Tool("diffuse-light", "Diffuse sky reference", "Light", 1, (
        number("sectors", "Directional slices", 8, 1),
        number("dark_level", "Dark reference", 0, .005),
        number("white_level", "Bright reference", 1, .005),
    ), "Compare relative camera luminance from open sky with angle weighting for a horizontal plane.", diffuse_reference, True),
    Tool("paired-change", "Repeat survey change", "Fieldwork", 2, (
        number("difference_floor", "Luminance change indicator", .08, .005),
    ), "Look for open-to-obstructed changes in matched directions on two repeat photographs.", refind_change, True),
    Tool("repeat-align", "Repeat wall-corner alignment", "Fieldwork", 2, (
        structured("corners", "Matching unchanged wall corners", DEMO_CORNERS,
                   "Each entry needs first/second image points, a near or far band, and support."),
        number("difference_floor", "Luminance change indicator", .08, .005),
        number("marking_radius_px", "Corner marking radius (frame px)", 6, .5),
        number("trust_radius_fraction", "Trust radius (fraction of width)", .30, .01),
    ), "Register the repeats on unchanged wall corners, keep the untouched comparison, and only trust changes near a trusted corner.", wall_registration),
    Tool("direction-transect", "Direction transect", "Fieldwork", 1, (
        structured("from_direction", "Start bearing, elevation", [2, 10], shape="pair"),
        structured("to_direction", "End bearing, elevation", [85, 60], shape="pair"),
        number("positions", "Positions along the arc", 121, 1),
    ), "Trace a sky arc and read where it hits the source image, including blank sections.", transect),
    Tool("perspective", "Look toward bearing", "Camera", 1, (
        number("bearing", "Camera view heading (°)", 120, .1),
        number("elevation", "Elevation (°)", 29, .1),
        number("horizontal", "Width of view (°)", 85, .1),
        number("vertical", "Height of view (°)", 54, .1),
    ), "Inspect a slice of the fisheye with the image in ordinary perspective and a compass guide.", virtual_view),
    Tool("field-detail", "Local image detail", "Camera", 1, (
        number("columns", "Columns", 8, 1),
        number("rows", "Rows", 6, 1),
        number("edge_floor", "Detail threshold", .023, .001),
    ), "Check native detail, gradient direction and missing support by source region.", field_detail),
    Tool("color-check", "Color model validation", "Canopy", 1, (
        structured("regions", "Independent validation regions",
             [{"label": "Center sky", "kind": "sky", "vertices": [[.43,.43],[.55,.43],[.55,.55],[.43,.55]]}]),
    ), "Use a labelled check patch to see where planned camera classification agrees or needs review.", check_samples),
    Tool("camera-window", "Camera evidence windows", "Camera", 1, (
        structured("windows", "Windows on the source",
              [{"title":"Inner canopy","corners":[[.35,.35],[.62,.62]]}], "Use image coordinates between 0 and 1."),
        number("detail_window", "Enlarge window", 1, 1),
    ), "Read native pixels and camera direction for one or more inspection windows.", native_window),
    Tool("tie-audit", "Repeat landmark agreement", "Fieldwork", 2, (
        structured("pairs", "Corresponding marks", DEMO_TIES),
        number("tolerance", "Agreement tolerance (°)", 3, .1),
        number("neighborhood", "Marking radius (px)", 4, .25),
    ), "Check that individually marked landmarks agree in angular coordinates across two captures.", compare_ties),
    Tool("lens-audit", "Survey lens model check", "Fieldwork", 1, (
        structured("marks", "Native and survey landmarks", DEMO_OPTIC_MARKS),
        number("tail", "Residual quantile", .85, .01),
    ), "Compare where common radial lens models put the same independent landmark positions.", lens_model_audit),
    Tool("exposure-audit", "Repeat brightness references", "Light", 2, (
        structured("patches", "Directional reference patches", DEMO_BRIGHTNESS),
    ), "Check common camera coverage and observed brightness ratios for repeated angular reference patches.", relative_light_audit),
    Tool("surface-orientation", "Oriented surface sky view", "Light", 1, (
        structured("surfaces", "Surface directions", DEMO_PLANES, "Each normal is a bearing/elevation pair."),
        number("indicated_bearing", "Start bearing of sector table", 0, 1),
    ), "Compare available sky from differently oriented surfaces at the same station.", oriented_surfaces, True),
    Tool("field-marks", "Field marks report", "Fieldwork", 1, (
        structured("marks", "Field landmark notes",
                   [{"title": "Survey north", "direction": [0, 25], "role": "reference"},
                    {"title": "Upper canopy", "direction": [48, 68], "role": "opening"}]),
        number("tolerance", "Agreement tolerance (°)", 3, .1),
        number("origin", "Sort clockwise from bearing (°)", 0, 1)
    ), "Read native and declared directions for field notes, with measured support and residuals.", field_overlay),
)}
