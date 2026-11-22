"""Direct opening along a supplied path and native angular transects."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from .bench import Bench, Source
from .camera import rgb_luminance, srgb_to_linear
from .contracts import Problem, angle, fields, items, mapping, number, real, timestamp, whole
from .projection import angular_distance, interpolate_arc
from .result import Result
from .sky import LABELS, MISSING, OCCLUDED, OPEN, UNCERTAIN
from .visual import GOLD, MINT, PINK, height_chart, sun_path_on_photo


@dataclass(frozen=True)
class PathMark:
    moment: datetime
    direction: tuple[float, float]

    @classmethod
    def parse(cls, raw: dict) -> PathMark:
        doc = mapping(raw, "Path landmark")
        fields(doc, "at", "direction")
        return cls(datetime.fromisoformat(timestamp(doc.get("at"), "Path point time")), angle(doc.get("direction")))

    def public(self) -> dict:
        return {"at": self.moment.isoformat(timespec="seconds"), "direction": list(self.direction)}


DEMO_PATH = [
    {"at": "2026-06-09T07:30:00+00:00", "direction": [74, 14]},
    {"at": "2026-06-09T08:30:00+00:00", "direction": [92, 27]},
    {"at": "2026-06-09T09:30:00+00:00", "direction": [111, 39]},
    {"at": "2026-06-09T10:30:00+00:00", "direction": [138, 52]},
    {"at": "2026-06-09T11:30:00+00:00", "direction": [168, 55]},
    {"at": "2026-06-09T12:30:00+00:00", "direction": [202, 51]},
    {"at": "2026-06-09T13:30:00+00:00", "direction": [229, 42]},
    {"at": "2026-06-09T14:30:00+00:00", "direction": [248, 33]},
    {"at": "2026-06-09T15:30:00+00:00", "direction": [270, 20]},
]


@dataclass(frozen=True)
class PathIntervals:
    centres: np.ndarray
    starts: tuple[datetime, ...]
    ends: tuple[datetime, ...]
    anchors: np.ndarray

    @classmethod
    def sample(cls, values: tuple[PathMark, ...], seconds: int) -> PathIntervals:
        if (values[-1].moment - values[0].moment).total_seconds() > 49 * 3600:
            raise Problem("One direction path may cover at most 49 hours.")
        points = []
        starts = []
        stops = []
        sequence = []
        for i, (first, second) in enumerate(zip(values[:-1], values[1:], strict=True)):
            length_seconds = (second.moment - first.moment).total_seconds()
            if length_seconds <= 0:
                raise Problem("Path landmarks must be in strictly increasing time order.")
            pieces = int(np.ceil(length_seconds / seconds))
            edges = np.linspace(0, 1, pieces + 1)
            mids = (edges[:-1] + edges[1:]) / 2
            points.append(interpolate_arc(first.direction, second.direction, mids))
            starts.extend(first.moment + timedelta(seconds=length_seconds * float(a)) for a in edges[:-1])
            stops.extend(first.moment + timedelta(seconds=length_seconds * float(b)) for b in edges[1:])
            sequence.extend([i + 1] * pieces)
        if len(starts) > 2400:
            raise Problem("This time spacing produces over 2400 path probes.")
        return cls(np.concatenate(points), tuple(starts), tuple(stops), np.array(sequence))

    def durations(self) -> np.ndarray:
        return np.array([(a - b).total_seconds() for b, a in zip(self.starts, self.ends, strict=True)])


def _path_rows(intervals: PathIntervals, samples) -> list[dict]:
    rows = []
    for i, (a, b) in enumerate(zip(intervals.starts, intervals.ends, strict=True)):
        row = samples.readout(i)
        row["start_utc"] = a.isoformat(timespec="milliseconds")
        row["end_utc"] = b.isoformat(timespec="milliseconds")
        row["seconds"] = number((b - a).total_seconds(), 3)
        row["landmark_pair"] = int(intervals.anchors[i])
        rows.append(row)
    return rows


def _runs(readings: list[dict]) -> list[dict]:
    if not readings:
        return []
    summaries = []
    pending = None
    for row in readings:
        value = row["class"]
        if pending is None or pending["class"] != value:
            if pending is not None:
                summaries.append(pending)
            pending = {"class": value, "start_utc": row["start_utc"],
                       "end_utc": row["end_utc"], "seconds": row["seconds"], "probes": 1}
        else:
            pending["end_utc"] = row["end_utc"]
            pending["seconds"] = number(pending["seconds"] + row["seconds"], 3)
            pending["probes"] += 1
    if pending:
        summaries.append(pending)
    return summaries


def _class_duration(intervals: PathIntervals, codes: np.ndarray) -> dict:
    dwell = intervals.durations()
    counts = np.bincount(codes, weights=dwell, minlength=len(LABELS))
    unsettled = counts[MISSING] + counts[UNCERTAIN]
    return {"seconds_by_class": {label: number(duration, 3) for label, duration in zip(LABELS, counts, strict=True)},
            "total_seconds": number(dwell.sum(), 3),
            "open_time_bounds_minutes": [number(counts[OPEN] / 60, 4), number((counts[OPEN] + unsettled) / 60, 4)],
            "confirmed_time_share": number((counts[OPEN] + counts[OCCLUDED]) / dwell.sum(), 5)}


def direct_path(bench: Bench) -> Result:
    source = bench.one()
    data = bench.order.settings
    fields(data, "landmarks", "step")
    step = whole(data.get("step", 240), "Probe interval in seconds", 60, 3600)
    input_points = items(data.get("landmarks", DEMO_PATH), "Path landmarks", 2, 96)
    values = tuple(PathMark.parse(raw) for raw in input_points)
    intervals = PathIntervals.sample(values, step)
    sampled = source.sample(intervals.centres)
    raw = _path_rows(intervals, sampled)
    runs = _runs(raw)
    result = Result("sun-path", "Direct opening along a direction path",
                    "A supplied sequence of sky directions is checked against its camera evidence. Time with an uncertain or missing line of sight stays unresolved.")
    result.metrics = {**_class_duration(intervals, sampled.codes), "inspection_samples": len(raw),
                      "input_landmarks": [mark.public() for mark in values], "continuous_runs": len(runs),
                      "maximum_step_seconds": step}
    result.readings = runs
    result.png("path-on-source", "Path and supporting camera pixels",
               sun_path_on_photo(source.camera, sampled.native, sampled.codes, sampled.measured))
    result.png("path-elevation", "Time and class profile", _elevation_plot(intervals, sampled.codes))
    result.csv("class-intervals", "Contiguous class intervals", runs)
    result.csv("path-direction-samples", "Timed direction and source coordinates", raw)
    result.warnings = [*source.method_warnings(),
                       "Times and bearings come from the entered landmarks. No solar ephemeris or flux is inferred."]
    result.probe_kind = "direction"
    return result


def _elevation_plot(interval: PathIntervals, codes: np.ndarray):
    colors = (("Observed sky", MINT, OPEN), ("Observed obstruction", PINK, OCCLUDED),
              ("Unsettled or missing", GOLD, UNCERTAIN))
    elev = interval.centres[:, 1]
    tracks = []
    for name, color, val in colors:
        known = codes == val if val != UNCERTAIN else (codes == UNCERTAIN) | (codes == MISSING)
        line = [float(e) if k else None for e, k in zip(elev, known, strict=True)]
        tracks.append({"name": name, "color": color, "values": line})
    return height_chart(tracks, "Direction path by sample order", "Elevation in degrees",
                        "Earliest to latest ordered samples", hard_bounds=(0, 90))


def _arc_controls(data: dict):
    fields(data, "from_direction", "to_direction", "positions")
    a = angle(data.get("from_direction", [2, 10]), "Start direction")
    b = angle(data.get("to_direction", [85, 60]), "End direction")
    distance = float(angular_distance(a, b))
    if distance < .05:
        raise Problem("Separate the transect endpoints in the viewing sphere.")
    count = whole(data.get("positions", 121), "Angular positions", 10, 420)
    return a, b, distance, count


def transect(bench: Bench) -> Result:
    source = bench.one()
    a, b, distance, count = _arc_controls(bench.order.settings)
    steps = np.linspace(0, 1, count)
    positions = interpolate_arc(a, b, steps)
    sampled = source.sample(positions)
    luma = rgb_luminance(srgb_to_linear(sampled.rgb))
    rows = []
    for i, step in enumerate(steps):
        row = sampled.readout(i)
        row["degrees_from_start"] = number(distance * step)
        row["luminance"] = number(luma[i]) if sampled.measured[i] else None
        rows.append(row)
    report = Result("direction-transect", "Direction to source transect",
                    "A short great-circle path locates each sample in the saved source lens and leaves unmeasured positions blank.")
    report.metrics = {"arc_degrees": number(distance, 4), "positions": count,
                      "positions_with_camera_support": int(np.count_nonzero(sampled.measured)),
                      "positions_with_decisive_class": int(np.count_nonzero(sampled.known)),
                      "from_direction": a, "to_direction": b}
    report.readings = rows
    report.png("angular-transect", "Transect traced on original",
               sun_path_on_photo(source.camera, sampled.native, sampled.codes, sampled.measured))
    report.png("transect-signal", "Signal by angular distance", height_chart(
        [{"name": "Linear display luminance", "color": MINT, "values": [row["luminance"] for row in rows]}],
        "Direction transect", "Display luminance", f"{distance:.1f} degrees along arc"))
    report.csv("direction-lineage", "Angular plane to native sample table", rows)
    report.warnings = source.method_warnings()
    report.probe_kind = "direction"
    return report
