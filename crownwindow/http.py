"""Local HTTP entry points for cameras, evidence and the observation workbench."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from threading import Lock

import numpy as np
from flask import Flask, Response, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from .archive import MAX_PACKET_COMPRESSED, build_packet, render_sheet, verify_packet
from .bench import ANALYSIS_VERSION, Bench, Capacity, Order, Source
from .calibration import fit_lens, lens_limits
from .camera import FORMATS, MAX_FILE_BYTES, MAX_PIXELS, Camera, encode_png, safe_download_label
from .contracts import (Capture, Mark, Problem, Station, boolean, enum, fields, filename, identifier,
                        items, key_of, mapping, point, real, text, whole)
from .demo import EXAMPLES, populate, render_observation
from .field import direction_readout, image_readout, native_witness
from .folio import contact_sheet, folio_csv, folio_name, make_folio, render_folio
from .library import MAX_MARKS_PER_OBSERVATION, MAX_OBSERVATIONS_PER_STATION, MAX_RECORDS_PER_STATION, Library
from .lean import (HOLDOUT_COLUMNS, HOLDOUT_ROWS, lean_fit, lean_marks, lens_with_holdout,
                   leveling_agreement, single_placement)
from .mark_exchange import MAX_NOTE_BYTES, field_log, import_log
from .projection import Lens, RasterHoldout
from .packet_reader import evidence_receipt_image, inspect_packet
from .result import Asset
from .sky import Classifier, Controls, native_classes
from .tools import TOOLS
from .trajectory_io import MAX_CSV, read_path_csv, write_path_csv
from .paths import DEMO_PATH
from .viewpoint import View, perspective_probe
from .visual import MINT, PINK, calibration_view, plot_over_camera, tinted_rgb


def _request() -> dict:
    if not request.is_json:
        raise Problem("Send this operation as a JSON request.")
    return mapping(request.get_json(silent=True), "Request")


def _part(key: str, limit: int) -> tuple[str, bytes]:
    part = request.files.get(key)
    if part is None or not part.filename:
        raise Problem("Choose a local file.")
    raw = part.read(limit + 1)
    if len(raw) > limit:
        raise Problem("The chosen file exceeds the import budget.", 413)
    if not raw:
        raise Problem("The chosen file is empty.")
    return part.filename, raw


def _json_form(key: str) -> dict:
    raw = request.form.get(key)
    if raw is None:
        return {}
    try:
        return mapping(json.loads(raw), key)
    except json.JSONDecodeError as error:
        raise Problem(f"The {key} information was not valid JSON.") from error


def _send(contents: bytes, mime: str, name: str, *, attachment: bool = False):
    result = send_file(io.BytesIO(contents), mimetype=mime, download_name=filename(name),
                       as_attachment=attachment, max_age=0)
    result.headers["X-Content-Type-Options"] = "nosniff"
    return result


def _preview(name: str, label: str, picture) -> dict:
    return Asset(name, "image/png", encode_png(picture), label).inline()


def _stored_source(library: Library, photo_key: str, override: dict | None = None) -> Source:
    key = identifier(photo_key, "Observation")
    photo = library.photo(key)
    _, rows = library.acquire(photo["station"], [key], override or {})
    return Source.from_stored(*rows[0])


def _retained_source(library: Library, source: dict) -> Source:
    photo = {
        "id": source["id"], "capture": source["capture"],
        "image": source["image"], "classification": source["classification"],
        "original": source["original"], "preview": source["preview"],
        "revision": source["source_revision"],
    }
    lens = None
    if source.get("lens_id") is not None:
        lens = {"id": source["lens_id"], "lens": source["lens"], "title": source["lens_title"]}
    return Source.from_stored(photo, lens, library.source_blob(source))


def register_library(app: Flask, library: Library, executor: Capacity):
    open_example = Lock()

    @app.get("/api/stations")
    def stations():
        return {"stations": library.stations()}

    @app.post("/api/stations")
    def create_station():
        station, _ = library.create_station(Station.parse(_request()))
        return {"station": station}, 201

    @app.get("/api/stations/<key>")
    def station_details(key):
        return library.station_details(key)

    @app.put("/api/stations/<key>")
    def edit_station(key):
        data = _request()
        fields(data, "revision", "station")
        revision = whole(data.get("revision"), "Station revision", 1, 2_000_000)
        values = Station.parse(data.get("station"))
        return {"station": library.edit_station(key, revision, values)}

    @app.get("/api/examples")
    def examples():
        return {"examples": [example.public() for example in EXAMPLES.values()]}

    @app.get("/api/examples/<key>/preview")
    def example_image(key):
        if key not in EXAMPLES:
            raise Problem("That example is unavailable.", 404)
        with executor:
            image = render_observation(key, False, True)
        return _send(image, "image/png", key + ".png")

    @app.post("/api/examples/<key>")
    def load_example(key):
        if key not in EXAMPLES:
            raise Problem("That example is unavailable.", 404)
        with executor, open_example:
            station, created = populate(library, key)
        return {"station": station, "created": created}, 201 if created else 200

    @app.post("/api/stations/<station>/photos")
    def load_camera(station):
        name, raw = _part("image", MAX_FILE_BYTES)
        capture = Capture.parse(_json_form("capture"), fallback=Path(name).stem)
        with executor:
            image = Camera.read(raw, name)
            photo = library.add_photo(station, image, raw, capture)
        return {"photo": photo}, 201

    @app.get("/api/photos/<key>")
    def photo(key):
        return {"photo": library.photo(key)}

    @app.put("/api/photos/<key>")
    def edit_photo(key):
        data = _request()
        fields(data, "revision", "capture")
        version = whole(data.get("revision"), "Observation revision", 1, 2_000_000)
        return {"photo": library.edit_photo(key, version, Capture.parse(data.get("capture")))}

    @app.get("/api/photos/<key>/original")
    def original(key):
        photo, raw = library.photo_bytes(key)
        return _send(raw, photo["image"]["mime"], photo["image"]["file"], attachment=True)

    @app.get("/api/photos/<key>/preview")
    def photo_preview(key):
        photo, raw = library.photo_bytes(key)
        with executor:
            camera = Camera.read(raw, photo["image"]["file"])
            image = encode_png(camera.code_preview())
        return _send(image, "image/png", safe_download_label(camera) + ".png")


def _check_version(source: Source, version: object) -> int:
    revision = whole(version, "Observation revision", 1, 2_000_000)
    if revision != source.record["revision"]:
        raise Problem("The observation changed. Reopen it before applying another edit.", 409)
    return revision


def register_calibration(app: Flask, library: Library, executor: Capacity):
    @app.get("/api/photos/<key>/lens")
    def lens_versions(key):
        return library.lens_records(key)

    @app.post("/api/photos/<key>/lens")
    def lens_preview(key):
        data = _request()
        fields(data, "revision", "title", "lens", "marks", "retain", "expected")
        name = text(data.get("title"), "Lens version title", 78, default="Lens placement")
        requested = Lens.parse(data.get("lens"))
        marks = items(data.get("marks", []), "Landmarks", 0, 30)
        keep = boolean(data.get("retain", False), "Retain lens")
        with executor:
            source = _stored_source(library, key)
            revision = _check_version(source, data.get("revision"))
            if marks:
                fit = fit_lens(source.camera, requested, marks)
                lens = fit.lens
                proof = fit.public()
            else:
                lens = requested
                proof = {"mode": "manual placement", "metrics": {}}
            fingerprint = key_of({"title": name, "source": source.public(), "lens": requested.public(), "marks": marks})
            if keep and data.get("expected") != fingerprint:
                raise Problem("Review this exact lens placement before keeping it.", 409)
            images = _alignment_images(source, lens, proof.get("residuals", []))
            response = {"title": name, "revision": revision, "lens": lens.public(), "fit": proof,
                        "coverage": lens_limits(lens, source.camera), "fingerprint": fingerprint}
            response["assets"] = images
            if keep:
                saved = library.add_lens(key, revision, name, lens, proof)
                response["saved"] = saved
                response["revision"] = library.photo(key)["revision"]
                return response, 201
        return response

    @app.put("/api/photos/<key>/active-lens")
    def activate_lens(key):
        data = _request()
        fields(data, "revision", "lens")
        version = whole(data.get("revision"), "Observation revision", 1, 2_000_000)
        lens = None if data.get("lens") is None else identifier(data.get("lens"), "Lens")
        return {"photo": library.switch_lens(key, version, lens)}

    @app.post("/api/photos/<key>/leveling")
    def leveling_preview(key):
        data = _request()
        fields(data, "revision", "lens", "marks", "threshold")
        with executor:
            source = _stored_source(library, key)
            revision = _check_version(source, data.get("revision"))
            template = Lens.parse(data.get("lens"))
            marks = lean_marks(data.get("marks"))
            threshold = real(data.get("threshold", 2.5), "Placement disagreement threshold", .1, 12)
            fixed, fixed_fit, fixed_warnings = lean_fit(source.camera, template, marks, "fixed_circle")
            try:
                adjusted, adjusted_fit, adjusted_warnings = lean_fit(source.camera, template, marks, "adjusted_circle")
            except Problem as error:
                adjusted = adjusted_fit = adjusted_warnings = None
                adjusted_error = str(error)
            if adjusted is None:
                analysis = single_placement(source.camera, fixed, fixed_fit, fixed_warnings, adjusted_error)
            else:
                analysis = leveling_agreement(source.camera, fixed, adjusted, fixed_fit, adjusted_fit,
                                              fixed_warnings, adjusted_warnings, threshold)
            image_titles = (("panel", "leveling-placement.png", "Both placements over the original"),
                            ("angular", "leveling-agreement.png", "Concordant and withheld directions"),
                            ("native", "leveling-on-source.png", "Agreement painted on the camera"))
            assets = [_preview(name, label, analysis["images"][key])
                      for key, name, label in image_titles if analysis["images"][key] is not None]
            payload = {k: v for k, v in analysis.items() if k != "images"}
            payload.update({"revision": revision, "assets": assets,
                            "fingerprint": key_of({"template": template.public(), "marks": [m.public() for m in marks],
                                                   "threshold": threshold, "analysis": {
                                                       "fits": payload["fits"], "holdouts": payload["holdouts"]}})})
        return payload

    @app.post("/api/photos/<key>/leveling/retain")
    def leveling_retain(key):
        data = _request()
        fields(data, "revision", "title", "mode", "holdout", "holdout_data", "lens",
               "placements", "retain", "expected", "fingerprint")
        name = text(data.get("title"), "Leveled lens title", 78, default="Leveled placement")
        keep = boolean(data.get("retain", False), "Retain lens")
        available = tuple(items(data.get("placements", []), "Available placements", 1, 2))
        chosen = enum(data.get("mode"), "Leveled placement", ("fixed_circle", "adjusted_circle"))
        if chosen not in available:
            raise Problem("That placement is not available for the current corner set.")
        holdout_kind = enum(data.get("holdout"), "Withheld belt", ("none", "disputed", "cautious"))
        with executor:
            source = _stored_source(library, key)
            revision = _check_version(source, data.get("revision"))
            lens = Lens.parse(data.get("lens"))
            if not lens.tilted:
                raise Problem("Retain a leveled placement that recovered a non-zero optical-axis tilt.")
            holdout = None
            if holdout_kind != "none":
                raster = RasterHoldout.parse(data.get("holdout_data"))
                if raster is None or raster.columns != HOLDOUT_COLUMNS or raster.rows != HOLDOUT_ROWS:
                    raise Problem("The withheld belt must come from a current leveling preview.")
                holdout = raster
            lens = lens_with_holdout(lens, holdout)
            proof = {"mode": "leveling", "placement": chosen, "holdout": holdout_kind,
                     "holdout_cells": holdout.cell_count() if holdout else 0,
                     "preview_fingerprint": data.get("fingerprint")}
            fingerprint = key_of({"title": name, "source": source.public(), "lens": lens.public()})
            if keep and data.get("expected") != fingerprint:
                raise Problem("Review this exact leveled placement before keeping it.", 409)
            images = _alignment_images(source, lens, [])
            response = {"title": name, "revision": revision, "lens": lens.public(), "fit": proof,
                        "coverage": lens_limits(lens, source.camera), "fingerprint": fingerprint,
                        "assets": images}
            if keep:
                saved = library.add_lens(key, revision, name, lens, proof)
                response["saved"] = saved
                response["revision"] = library.photo(key)["revision"]
                return response, 201
        return response


def _alignment_images(source: Source, lens: Lens, residuals: list[dict]) -> list[dict]:
    chart = calibration_view(source.camera, lens)
    images = [_preview("lens-on-source.png", "Camera and field directions", chart)]
    if residuals:
        points = []
        for row in residuals:
            u, v = row["entered_image"]
            color = PINK if row["review_needed"] else MINT
            points.append((u, v, row["mark"][:20] + f" {row['residual_px']:g}px", color))
        images.append(_preview("landmark-residuals.png", "Marked points and residuals",
                               plot_over_camera(source.camera, points)))
    return images


def register_marks(app: Flask, library: Library, executor: Capacity):
    @app.get("/api/photos/<key>/marks")
    def list_marks(key):
        return {"marks": library.marks(key)}

    @app.post("/api/photos/<key>/marks")
    def add_mark(key):
        data = _request()
        fields(data, "revision", "mark")
        version = whole(data.get("revision"), "Observation revision", 1, 2_000_000)
        return {"mark": library.add_mark(key, version, Mark.parse(data.get("mark")))}, 201

    @app.get("/api/photos/<key>/mark-log/<fmt>")
    def export_notes(key, fmt):
        allowed = enum(fmt, "Field log format", ("json", "csv"))
        data = field_log(library, key, csv_only=allowed == "csv")
        mime = "text/csv; charset=utf-8" if allowed == "csv" else "application/json"
        return _send(data, mime, "field-marks." + allowed, attachment=True)

    @app.post("/api/photos/<key>/mark-log")
    def read_notes(key):
        _, raw = _part("notes", MAX_NOTE_BYTES)
        try:
            revision = int(request.form.get("revision", ""))
        except ValueError as error:
            raise Problem("This notes import requires the displayed source revision.") from error
        mode = enum(request.form.get("stage", "preview"), "Field log stage", ("preview", "retain"))
        return import_log(library, key, revision, raw, commit=mode == "retain")

    @app.post("/api/photos/<key>/classify")
    def preview_color(key):
        data = _request()
        fields(data, "revision", "classifier", "retain", "expected")
        new = Controls.parse(data.get("classifier"))
        keep = boolean(data.get("retain", False), "Retain controls")
        with executor:
            source = _stored_source(library, key)
            revision = _check_version(source, data.get("revision"))
            check = key_of({"source": source.public(), "classifier": new.public()})
            if keep and data.get("expected") != check:
                raise Problem("Preview these exact color samples before saving.", 409)
            trained = Classifier.train(source.camera, source.lens, new)
            rgb, labels, _ = native_classes(source.camera, source.lens, trained)
            image = tinted_rgb(rgb, labels)
            output = {
                "fingerprint": check, "classifier": trained.describe(),
                "preview": _preview("trial-colors.png", "Trial classification", image),
            }
            if keep:
                output["photo"] = library.set_controls(key, revision, new)
                return output, 201
        return output


def _probe(source: Source, data: dict) -> list[dict]:
    mode = enum(data.get("space", "direction"), "Probe coordinate system", ("direction", "image"))
    locations = data.get("positions")
    if mode == "direction":
        return direction_readout(source, locations)
    return image_readout(source, locations, real(data.get("radius", 3), "Neighbourhood in pixels", 1, 32))


def _probe_result(source: Source, data: dict) -> dict:
    values = _probe(source, data)
    result = {"source": source.public(), "readings": values}
    if boolean(data.get("witness", False), "Return pixel witness"):
        if len(values) != 1:
            raise Problem("Choose one position for the magnified source evidence.")
        radius = real(data.get("radius", 6), "Pixel witness radius", 1, 32)
        result["witness"] = _preview("native-evidence.png", "Exact native sample neighbourhood",
                                     native_witness(source, values[0], radius))
    return result


def register_analysis(app: Flask, library: Library, executor: Capacity):
    @app.get("/api/tools")
    def tools():
        return {"tools": [entry.public() for entry in TOOLS.values()]}

    @app.post("/api/photos/<key>/probe")
    def live_probe(key):
        data = _request()
        fields(data, "space", "positions", "lens", "radius", "witness")
        override = {key: data["lens"]} if "lens" in data else {}
        with executor:
            source = _stored_source(library, key, override)
            return _probe_result(source, data)

    @app.post("/api/stations/<station>/tools/<kind>")
    def inspection(station, kind):
        tool = TOOLS.get(kind)
        if tool is None:
            raise Problem("That observation tool is unavailable.", 404)
        order = Order.parse(_request())
        with executor:
            if len(order.photos) != tool.photos:
                raise Problem(f"Select {tool.photos} observation{'s' if tool.photos != 1 else ''} for this tool.")
            bench = Bench.load(library, station, order)
            signature = bench.check_preview(kind)
            result = tool.run(bench)
            report = {
                **result.outline(),
                "order": order.public(),
                "sources": bench.make_source_identity(),
                "fingerprint": signature,
                "measurement_version": ANALYSIS_VERSION,
            }
            if order.retain:
                record, created = library.save_report(station, order.name, kind, signature, report, result.assets)
                return library.retained(record["id"]), 201 if created else 200
            report["assets"] = [media.inline() for media in result.assets]
        return report

    @app.get("/api/retained/<key>")
    def retained(key):
        return library.retained(key)

    @app.post("/api/retained/<key>/probe")
    def record_probe(key):
        data = _request()
        fields(data, "space", "positions", "source_index", "radius", "witness")
        with executor:
            record = library.retained(key)
            position = whole(data.get("source_index", 0), "Retained source", 0, len(record["sources"]) - 1)
            source = _retained_source(library, record["sources"][position])
            return _probe_result(source, data)

    @app.post("/api/stations/<station>/perspective-probe")
    def view_probe(station):
        data = _request()
        fields(data, "position", "view", "photo", "record")
        positions = [point(value, "Perspective point") for value in items(data.get("position"), "Perspective points", 1, 60)]
        with executor:
            if data.get("record"):
                saved = library.retained(data["record"])
                if saved["station"] != station or saved["tool"] != "perspective":
                    raise Problem("Open a perspective result from this station.", 409)
                viewpoint = View.parse(saved["order"]["settings"])
                source = _retained_source(library, saved["sources"][0])
            else:
                source = _stored_source(library, data.get("photo"))
                if source.record["station"] != station:
                    raise Problem("The source observation belongs to another station.", 409)
                viewpoint = View.parse(data.get("view"))
            response = perspective_probe(source, viewpoint, positions)
        return {"readings": response, "source": source.public(), "view": viewpoint.public()}

    @app.get("/api/retained/<key>/assets/<name>")
    def result_asset(key, name):
        info, data = library.read_asset(key, name)
        download = not info["mime"].startswith("image/")
        return _send(data, info["mime"], name, attachment=download)

    @app.get("/api/retained/<key>/sources")
    def compared_sources(key):
        return library.source_audit(key)

    @app.post("/api/retained/<key>/reviews")
    def review(key):
        body = _request()
        fields(body, "name", "status", "note", "seen_reviews")
        result = library.add_review(key, text(body.get("name"), "Reviewer", 75),
                                    enum(body.get("status"), "Review status", ("observed", "confirmed", "needs_revisit")),
                                    text(body.get("note"), "Review note", 500, default=""),
                                    whole(body.get("seen_reviews"), "Visible revision count", 0, 60))
        return {"review": result}, 201


def register_archives(app: Flask, library: Library, executor: Capacity):
    @app.get("/api/retained/<key>/packet")
    def bundle(key):
        include_originals = request.args.get("originals", "1") != "0"
        report = library.retained(key)
        with executor:
            label, raw = build_packet(library, report, include_originals)
        return _send(raw, "application/zip", label, attachment=True)

    @app.get("/api/retained/<key>/sheet")
    def sheet(key):
        return Response(render_sheet(library.retained(key)), content_type="text/html; charset=utf-8")

    @app.post("/api/packets/verify")
    def verify():
        _, raw = _part("packet", MAX_PACKET_COMPRESSED)
        with executor:
            return verify_packet(raw)


    @app.post("/api/packets/view")
    def portable_view():
        _, raw = _part("packet", MAX_PACKET_COMPRESSED)
        with executor:
            return inspect_packet(raw)

    @app.post("/api/packets/receipt")
    def packet_receipt():
        _, raw = _part("packet", MAX_PACKET_COMPRESSED)
        with executor:
            document = inspect_packet(raw)
            image = evidence_receipt_image(document)
        return _send(image, "image/png", "evidence-receipt.png", attachment=True)


def register_field_exchange(app: Flask, library: Library, executor: Capacity):
    @app.post("/api/paths/import")
    def import_path():
        name, raw = _part("directions", MAX_CSV)
        return read_path_csv(raw, name)

    @app.get("/api/paths/example.csv")
    def direction_example():
        return _send(write_path_csv(DEMO_PATH), "text/csv; charset=utf-8",
                     "example-direction-path.csv", attachment=True)

    @app.post("/api/stations/<station>/folio")
    def preview_folio(station):
        data = _request()
        fields(data, "reports", "title")
        document, _ = make_folio(library, station, data.get("reports"), data.get("title", ""))
        return document

    @app.get("/api/stations/<station>/folio/<kind>")
    def export_folio(station, kind):
        selected = request.args.getlist("report") or None
        document, reports = make_folio(library, station, selected, request.args.get("title", ""))
        if kind == "index.csv":
            return _send(folio_csv(document), "text/csv; charset=utf-8", folio_name(document, ".csv"), attachment=True)
        if kind == "contact.png":
            with executor:
                image = contact_sheet(library, document, reports)
            return _send(image, "image/png", folio_name(document, ".png"))
        if kind == "print.html":
            return Response(render_folio(document), content_type="text/html; charset=utf-8")
        raise Problem("That field folio export is unavailable.", 404)


def create_app(location: str | Path | None = None) -> Flask:
    static_dir = Path(__file__).resolve().parents[1] / "static"
    app = Flask(__name__, static_folder=str(static_dir), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_PACKET_COMPRESSED + 1_000_000
    directory = Path(location or os.environ.get("CROWNWINDOW_DATA", Path.cwd() / "data")).expanduser()
    library = Library(directory)
    executor = Capacity(2)
    app.extensions["crownwindow_library"] = library

    @app.errorhandler(Problem)
    def handled(error):
        return {"error": str(error)}, error.status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return {"error": "The request exceeds the local file budget."}, 413

    @app.get("/api/budget")
    def budget():
        return {
            "camera_mime": list(FORMATS.values()),
            "camera_file_bytes": MAX_FILE_BYTES,
            "camera_max_pixels": MAX_PIXELS,
            "field_note_file_bytes": MAX_NOTE_BYTES,
            "observations_per_station": MAX_OBSERVATIONS_PER_STATION,
            "retained_reports_per_station": MAX_RECORDS_PER_STATION,
            "field_marks_per_observation": MAX_MARKS_PER_OBSERVATION,
            "portable_packet_bytes": MAX_PACKET_COMPRESSED,
        }

    @app.get("/api/health")
    def health():
        return {"status": "ready", "app": "CrownWindow", "version": ANALYSIS_VERSION}

    @app.get("/")
    def home():
        return app.send_static_file("index.html")

    register_library(app, library, executor)
    register_calibration(app, library, executor)
    register_marks(app, library, executor)
    register_analysis(app, library, executor)
    register_archives(app, library, executor)
    register_field_exchange(app, library, executor)
    return app
