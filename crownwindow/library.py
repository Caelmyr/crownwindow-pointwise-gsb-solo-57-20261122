"""SQLite observation library and content-addressed, immutable evidence."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import os
import sqlite3
import uuid

from .camera import Camera
from .contracts import Capture, Mark, Problem, Station, canonical, identifier, sha, token, utc_now
from .projection import Lens
from .sky import Controls


SCHEMA = """
CREATE TABLE IF NOT EXISTS station (
    key TEXT PRIMARY KEY,
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    revision INTEGER NOT NULL,
    document TEXT NOT NULL,
    demo_key TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS photograph (
    key TEXT PRIMARY KEY,
    station TEXT NOT NULL REFERENCES station(key),
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    revision INTEGER NOT NULL,
    blob TEXT NOT NULL,
    image TEXT NOT NULL,
    document TEXT NOT NULL,
    active_lens TEXT,
    controls TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lens_record (
    key TEXT PRIMARY KEY,
    photograph TEXT NOT NULL REFERENCES photograph(key),
    ordinal INTEGER NOT NULL,
    created TEXT NOT NULL,
    title TEXT NOT NULL,
    lens TEXT NOT NULL,
    report TEXT NOT NULL,
    UNIQUE(photograph, ordinal)
);
CREATE TABLE IF NOT EXISTS field_mark (
    key TEXT PRIMARY KEY,
    photograph TEXT NOT NULL REFERENCES photograph(key),
    created TEXT NOT NULL,
    document TEXT NOT NULL,
    lens_key TEXT REFERENCES lens_record(key)
);
CREATE TABLE IF NOT EXISTS retained (
    key TEXT PRIMARY KEY,
    station TEXT NOT NULL REFERENCES station(key),
    created TEXT NOT NULL,
    name TEXT NOT NULL,
    tool TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    document TEXT NOT NULL,
    UNIQUE(station, name, fingerprint)
);
CREATE TABLE IF NOT EXISTS retained_asset (
    retained TEXT NOT NULL REFERENCES retained(key),
    name TEXT NOT NULL,
    blob TEXT NOT NULL,
    mime TEXT NOT NULL,
    length INTEGER NOT NULL,
    PRIMARY KEY(retained, name)
);
CREATE TABLE IF NOT EXISTS review_event (
    key TEXT PRIMARY KEY,
    retained TEXT NOT NULL REFERENCES retained(key),
    ordinal INTEGER NOT NULL,
    created TEXT NOT NULL,
    person TEXT NOT NULL,
    status TEXT NOT NULL,
    note TEXT NOT NULL,
    UNIQUE(retained, ordinal)
);
CREATE INDEX IF NOT EXISTS photo_station ON photograph(station, created);
CREATE INDEX IF NOT EXISTS marks_photo ON field_mark(photograph, created);
CREATE INDEX IF NOT EXISTS retained_station ON retained(station, created DESC);
"""


MAX_STATIONS = 250
MAX_OBSERVATIONS_PER_STATION = 28
MAX_MARKS_PER_OBSERVATION = 220
MAX_LENS_REVISIONS = 80
MAX_RECORDS_PER_STATION = 120
MAX_ASSET_BYTES = 28 * 1024 * 1024
MAX_TOTAL_RECORD_BYTES = 62 * 1024 * 1024


def _key() -> str:
    return uuid.uuid4().hex


def _json(data: dict | list) -> str:
    return canonical(data).decode("utf-8")


def _decode(value: str | None, fallback=None):
    return json.loads(value) if value else fallback


class Blobs:
    """Write content to a temporary file and atomically adopt its hash."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def path(self, digest: str) -> Path:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise Problem("Invalid local evidence address.", 500)
        return self.directory / digest[:2] / digest[2:]

    def add(self, value: bytes) -> str:
        digest = sha(value)
        target = self.path(digest)
        if target.exists():
            return digest
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name("." + target.name + "." + _key())
        try:
            with temporary.open("xb") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return digest

    def read(self, digest: str, *, verify: bool = False) -> bytes:
        try:
            data = self.path(digest).read_bytes()
        except FileNotFoundError as error:
            raise Problem("The local evidence file is missing.", 410) from error
        if verify and sha(data) != digest:
            raise Problem("A local evidence file no longer matches its recorded digest.", 409)
        return data


class Library:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.database = directory / "survey.sqlite"
        with self.session() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
        self.blobs = Blobs(directory / "blobs")

    @contextmanager
    def session(self, *, write: bool = False):
        connection = sqlite3.connect(self.database, isolation_level=None, timeout=13)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if write:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _require(db: sqlite3.Connection, sql: str, key: str, unavailable: str) -> sqlite3.Row:
        row = db.execute(sql, (identifier(key),)).fetchone()
        if row is None:
            raise Problem(unavailable, 404)
        return row

    @staticmethod
    def _station(row: sqlite3.Row) -> dict:
        return {"id": row["key"], "created_at": row["created"], "updated_at": row["updated"],
                "revision": row["revision"], "sample": row["demo_key"], **_decode(row["document"])}

    def _station_row(self, db: sqlite3.Connection, key: str) -> sqlite3.Row:
        return self._require(db, "SELECT * FROM station WHERE key = ?", key, "This station is not in this library.")

    @staticmethod
    def _photo(row: sqlite3.Row) -> dict:
        photo = {
            "id": row["key"], "station": row["station"], "created_at": row["created"],
            "updated_at": row["updated"], "revision": row["revision"], "image": _decode(row["image"]),
            "capture": _decode(row["document"]), "active_lens": row["active_lens"],
            "classification": _decode(row["controls"]),
        }
        photo["preview"] = f"/api/photos/{photo['id']}/preview"
        photo["original"] = f"/api/photos/{photo['id']}/original"
        return photo

    def _photo_row(self, db: sqlite3.Connection, key: str) -> sqlite3.Row:
        return self._require(db, "SELECT * FROM photograph WHERE key = ?", key, "This observation is not in the library.")

    @staticmethod
    def _lens(row: sqlite3.Row) -> dict:
        return {
            "id": row["key"], "photo": row["photograph"], "ordinal": row["ordinal"],
            "created_at": row["created"], "title": row["title"], "lens": _decode(row["lens"]),
            "report": _decode(row["report"]),
        }

    def _lens_row(self, db: sqlite3.Connection, key: str) -> sqlite3.Row:
        return self._require(db, "SELECT * FROM lens_record WHERE key = ?", key, "This lens version is unavailable.")

    @staticmethod
    def _record_head(row: sqlite3.Row) -> dict:
        return {"id": row["key"], "station": row["station"], "created_at": row["created"],
                "name": row["name"], "tool": row["tool"], "fingerprint": row["fingerprint"]}

    def _record_row(self, db: sqlite3.Connection, key: str) -> sqlite3.Row:
        return self._require(db, "SELECT * FROM retained WHERE key = ?", key, "This retained result is unavailable.")

    def stations(self) -> list[dict]:
        with self.session() as db:
            rows = db.execute(
                """SELECT station.*,
                    (SELECT count(*) FROM photograph WHERE photograph.station = station.key) as photos,
                    (SELECT count(*) FROM retained WHERE retained.station = station.key) as reports
                   FROM station ORDER BY updated DESC, key LIMIT ?""", (MAX_STATIONS,)).fetchall()
        return [{**self._station(row), "observations": row["photos"], "reports": row["reports"]} for row in rows]

    def create_station(self, data: Station, demo_key: str | None = None) -> tuple[dict, bool]:
        with self.session(write=True) as db:
            if demo_key is not None:
                existing = db.execute("SELECT * FROM station WHERE demo_key=?", (demo_key,)).fetchone()
                if existing:
                    return self._station(existing), False
            if db.execute("SELECT count(*) FROM station").fetchone()[0] >= MAX_STATIONS:
                raise Problem("This local library has reached its station budget.", 409)
            stamp = utc_now()
            identifier_value = _key()
            db.execute("INSERT INTO station VALUES (?,?,?,?,?,?)",
                       (identifier_value, stamp, stamp, 1, _json(data.public()), demo_key))
            return self._station(self._station_row(db, identifier_value)), True

    def edit_station(self, key: str, revision: int, value: Station) -> dict:
        with self.session(write=True) as db:
            row = self._station_row(db, key)
            self._match_revision(row, revision, "station")
            before = _decode(row["document"])
            if before.get("synthetic") and not value.synthetic:
                raise Problem("A clearly labelled synthetic sample cannot be relabelled as field evidence.", 409)
            db.execute("UPDATE station SET updated=?, revision=revision+1, document=? WHERE key=?",
                       (utc_now(), _json(value.public()), key))
            return self._station(self._station_row(db, key))

    @staticmethod
    def _match_revision(row: sqlite3.Row, revision: int, subject: str) -> None:
        if row["revision"] != revision:
            raise Problem(f"The {subject} changed on another screen. Reload before saving.", 409)

    def station_details(self, key: str) -> dict:
        with self.session() as db:
            station = self._station(self._station_row(db, key))
            photos = db.execute("SELECT * FROM photograph WHERE station=? ORDER BY created, rowid", (key,)).fetchall()
            reports = db.execute("SELECT * FROM retained WHERE station=? ORDER BY created DESC, rowid DESC", (key,)).fetchall()
            return {
                "station": station,
                "photos": [self._photo(row) for row in photos],
                "reports": [self._record_head(row) for row in reports],
            }

    def add_photo(self, station: str, camera: Camera, raw: bytes, capture: Capture) -> dict:
        with self.session(write=True) as db:
            self._station_row(db, station)
            current = db.execute("SELECT count(*) FROM photograph WHERE station=?", (station,)).fetchone()[0]
            if current >= MAX_OBSERVATIONS_PER_STATION:
                raise Problem("This station has reached its observation budget.", 409)
            stamp = utc_now()
            key = _key()
            digest = self.blobs.add(raw)
            if digest != camera.content_sha:
                raise Problem("Camera bytes changed during import.", 409)
            db.execute("INSERT INTO photograph VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (key, station, stamp, stamp, 1, digest, _json(camera.public()),
                        _json(capture.public()), None, _json(Controls.parse(None).public())))
            db.execute("UPDATE station SET updated=? WHERE key=?", (stamp, station))
            return self._photo(self._photo_row(db, key))

    def photo(self, key: str) -> dict:
        with self.session() as db:
            return self._photo(self._photo_row(db, key))

    def edit_photo(self, key: str, revision: int, data: Capture) -> dict:
        return self._edit_photo_column(key, revision, "document", _json(data.public()))

    def set_controls(self, key: str, revision: int, data: Controls) -> dict:
        return self._edit_photo_column(key, revision, "controls", _json(data.public()))

    def _edit_photo_column(self, key: str, revision: int, column: str, value: str) -> dict:
        if column not in ("document", "controls"):
            raise ValueError("Unsupported observation column")
        with self.session(write=True) as db:
            row = self._photo_row(db, key)
            self._match_revision(row, revision, "observation")
            stamp = utc_now()
            db.execute(f"UPDATE photograph SET {column}=?,updated=?,revision=revision+1 WHERE key=?",
                       (value, stamp, key))
            db.execute("UPDATE station SET updated=? WHERE key=?", (stamp, row["station"]))
            return self._photo(self._photo_row(db, key))

    def lens_records(self, photo_key: str) -> dict:
        with self.session() as db:
            photo = self._photo_row(db, photo_key)
            rows = db.execute("SELECT * FROM lens_record WHERE photograph=? ORDER BY ordinal DESC", (photo_key,)).fetchall()
            return {"active": photo["active_lens"], "versions": [self._lens(row) for row in rows]}

    def add_lens(self, photo_key: str, revision: int, title: str, lens: Lens, report: dict) -> dict:
        with self.session(write=True) as db:
            photo = self._photo_row(db, photo_key)
            self._match_revision(photo, revision, "observation")
            if photo["active_lens"]:
                current = self._lens_row(db, photo["active_lens"])
                if _decode(current["lens"]) == lens.public() and _decode(current["report"]) == report and current["title"] == title:
                    raise Problem("The selected lens and its report already match the latest version.", 409)
            existing = db.execute("SELECT coalesce(max(ordinal),0) FROM lens_record WHERE photograph=?",
                                  (photo_key,)).fetchone()[0]
            if existing >= MAX_LENS_REVISIONS:
                raise Problem("This observation has reached its lens history budget.", 409)
            key = _key()
            now = utc_now()
            db.execute("INSERT INTO lens_record VALUES (?,?,?,?,?,?,?)",
                       (key, photo_key, existing + 1, now, title, _json(lens.public()), _json(report)))
            self._switch_lens(db, photo, key)
            return self._lens(self._lens_row(db, key))

    def switch_lens(self, photo_key: str, revision: int, key: str | None) -> dict:
        with self.session(write=True) as db:
            photo = self._photo_row(db, photo_key)
            self._match_revision(photo, revision, "observation")
            if key is not None:
                lens = self._lens_row(db, key)
                if lens["photograph"] != photo_key:
                    raise Problem("That lens version belongs to a different observation.", 409)
            if photo["active_lens"] != key:
                self._switch_lens(db, photo, key)
            return self._photo(self._photo_row(db, photo_key))

    @staticmethod
    def _switch_lens(db: sqlite3.Connection, photo: sqlite3.Row, key: str | None) -> None:
        now = utc_now()
        db.execute("UPDATE photograph SET active_lens=?,updated=?,revision=revision+1 WHERE key=?",
                   (key, now, photo["key"]))
        db.execute("UPDATE station SET updated=? WHERE key=?", (now, photo["station"]))

    def photo_bytes(self, key: str) -> tuple[dict, bytes]:
        with self.session() as db:
            row = self._photo_row(db, key)
            return self._photo(row), self.blobs.read(row["blob"])

    def acquire(self, station_key: str, photo_keys: list[str], lens_overrides: dict[str, str | None]) -> tuple[dict, list[tuple[dict, dict | None, bytes]]]:
        if len(photo_keys) != len(set(photo_keys)):
            raise Problem("Select distinct observations.")
        if not set(lens_overrides).issubset(photo_keys):
            raise Problem("A lens override must refer to a selected observation.")
        with self.session() as db:
            db.execute("BEGIN")
            station = self._station(self._station_row(db, station_key))
            results = []
            for key in photo_keys:
                row = self._photo_row(db, key)
                if row["station"] != station_key:
                    raise Problem("Choose observations from the currently open station.", 409)
                lens_id = lens_overrides.get(key, row["active_lens"])
                lens = self._lens_row(db, lens_id) if lens_id else None
                if lens is not None and lens["photograph"] != key:
                    raise Problem("A lens override was saved for another observation.", 409)
                draft = (self._photo(row), self._lens(lens) if lens is not None else None,
                         self.blobs.read(row["blob"]))
                results.append(draft)
            db.commit()
        return station, results

    def marks(self, photo_key: str) -> list[dict]:
        with self.session() as db:
            self._photo_row(db, photo_key)
            rows = db.execute("SELECT * FROM field_mark WHERE photograph=? ORDER BY rowid", (photo_key,))
            return [{"id": row["key"], "photo": row["photograph"], "lens_id": row["lens_key"],
                     "created_at": row["created"], **_decode(row["document"])} for row in rows]

    def add_mark(self, photo_key: str, revision: int, mark: Mark) -> dict:
        return self.add_marks(photo_key, revision, [mark], deduplicate=False)[0]

    def add_marks(self, photo_key: str, revision: int, values: list[Mark], deduplicate: bool) -> list[dict]:
        with self.session(write=True) as db:
            photo = self._photo_row(db, photo_key)
            self._match_revision(photo, revision, "observation")
            existing = db.execute("SELECT document FROM field_mark WHERE photograph=?", (photo_key,)).fetchall()
            live = {_json(_decode(row["document"])) for row in existing}
            additions = []
            for mark in values:
                document = _json(mark.public())
                if deduplicate and document in live:
                    continue
                live.add(document)
                additions.append((mark, document))
            if len(existing) + len(additions) > MAX_MARKS_PER_OBSERVATION:
                raise Problem("This observation has reached its field mark budget.", 409)
            results = []
            stamp = utc_now()
            for mark, document in additions:
                key = _key()
                db.execute("INSERT INTO field_mark VALUES (?,?,?,?,?)",
                           (key, photo_key, stamp, document, photo["active_lens"]))
                results.append({"id": key, "photo": photo_key, "lens_id": photo["active_lens"],
                                "created_at": stamp, **mark.public()})
            return results

    def label_demo(self, station_key: str, sample: str) -> None:
        """Adopt the sample key after every demonstration source exists."""
        with self.session(write=True) as db:
            self._station_row(db, station_key)
            if db.execute("SELECT count(*) FROM photograph WHERE station=?", (station_key,)).fetchone()[0] == 0:
                raise Problem("A demonstration needs an image before it can be offered.", 409)
            db.execute("UPDATE station SET demo_key=? WHERE key=?", (sample, station_key))

    def retained(self, key: str) -> dict:
        with self.session() as db:
            row = self._record_row(db, key)
            record = _decode(row["document"])
            buttons = db.execute("SELECT * FROM retained_asset WHERE retained=? ORDER BY rowid", (key,)).fetchall()
            reviews = db.execute("SELECT * FROM review_event WHERE retained=? ORDER BY ordinal", (key,)).fetchall()
            return {**self._record_head(row), **record, "files": [self._asset_ref(x) for x in buttons],
                    "reviews": [self._review(x) for x in reviews]}

    @staticmethod
    def _asset_ref(row: sqlite3.Row) -> dict:
        return {"name": row["name"], "sha256": row["blob"], "mime": row["mime"], "size": row["length"],
                "url": f"/api/retained/{row['retained']}/assets/{row['name']}"}

    @staticmethod
    def _review(row: sqlite3.Row) -> dict:
        return {"id": row["key"], "ordinal": row["ordinal"], "created_at": row["created"],
                "reviewer": row["person"], "status": row["status"], "note": row["note"]}

    def source_blob(self, source: dict) -> bytes:
        digest = source["image"]["sha256"]
        return self.blobs.read(digest, verify=True)

    def source_audit(self, key: str) -> dict:
        record = self.retained(key)
        current = []
        with self.session() as db:
            for saved in record.get("sources", []):
                row = db.execute("SELECT * FROM photograph WHERE key=?", (saved["id"],)).fetchone()
                if row is None:
                    current.append({"id": saved["id"], "present": False, "saved": saved})
                    continue
                live = self._photo(row)
                current.append({
                    "id": saved["id"],
                    "present": True,
                    "title": live["capture"]["title"],
                    "source_bytes_match": live["image"]["sha256"] == saved["image"]["sha256"],
                    "capture_metadata_changed": live["capture"] != saved["capture"],
                    "current_classifier_changed": live["classification"] != saved["classification"],
                    "active_lens_changed": live["active_lens"] != saved.get("lens_id"),
                    "original": live["original"],
                    "saved": saved,
                    "current": live,
                })
        return {"id": key, "sources": current}

    def save_report(self, station: str, name: str, tool: str, fingerprint: str, document: dict, results: list) -> tuple[dict, bool]:
        total = sum(len(asset.data) for asset in results)
        if any(len(asset.data) > MAX_ASSET_BYTES for asset in results) or total > MAX_TOTAL_RECORD_BYTES:
            raise Problem("The result exceeds the retained-asset byte budget.", 413)
        with self.session(write=True) as db:
            self._station_row(db, station)
            existing = db.execute("SELECT * FROM retained WHERE station=? AND name=? AND fingerprint=?",
                                  (station, name, fingerprint)).fetchone()
            if existing:
                return self._record_head(existing), False
            count = db.execute("SELECT count(*) FROM retained WHERE station=?", (station,)).fetchone()[0]
            if count >= MAX_RECORDS_PER_STATION:
                raise Problem("This station has reached its retained result budget.", 409)
            key = _key()
            stamp = utc_now()
            db.execute("INSERT INTO retained VALUES (?,?,?,?,?,?,?)",
                       (key, station, stamp, name, tool, fingerprint, _json(document)))
            for asset in results:
                db.execute("INSERT INTO retained_asset VALUES (?,?,?,?,?)",
                           (key, token(asset.name), self.blobs.add(asset.data), asset.mime, len(asset.data)))
            db.execute("UPDATE station SET updated=? WHERE key=?", (stamp, station))
            return self._record_head(self._record_row(db, key)), True

    def read_asset(self, record: str, name: str) -> tuple[dict, bytes]:
        with self.session() as db:
            self._record_row(db, record)
            row = db.execute("SELECT * FROM retained_asset WHERE retained=? AND name=?",
                             (record, token(name))).fetchone()
            if row is None:
                raise Problem("That file was not retained with the report.", 404)
            return self._asset_ref(row), self.blobs.read(row["blob"], verify=True)

    def add_review(self, record: str, person: str, status: str, note: str, seen_count: int) -> dict:
        with self.session(write=True) as db:
            self._record_row(db, record)
            now = utc_now()
            key = _key()
            total = db.execute("SELECT count(*) FROM review_event WHERE retained=?", (record,)).fetchone()[0]
            if total != seen_count:
                raise Problem("Another review was appended. Reopen the report before adding yours.", 409)
            if total >= 60:
                raise Problem("This report reached its review history budget.", 409)
            db.execute("INSERT INTO review_event VALUES (?,?,?,?,?,?,?)",
                       (key, record, total + 1, now, person, status, note))
            return self._review(db.execute("SELECT * FROM review_event WHERE key=?", (key,)).fetchone())
