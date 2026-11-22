"use strict";

const main = document.querySelector("#main");
const dialog = document.querySelector("#dialog");
const notice = document.querySelector("#notice");
const state = {
  library: [], examples: [], tools: [], budget: {}, station: null, photos: [], reports: [],
  focus: null, selected: [], tab: "inspect", tool: "open-share", readings: null,
  settings: {}, orderSettings: null, image: null, activeAsset: null, probe: null, sourceIndex: 0,
  prepared: null, editor: null, marks: [], noLive: false, busy: false,
  lensPane: null, lensDraft: null, colorDraft: null, chosenColor: null, chosenLens: null, asked: null,
  uploads: {}, folio: null, toolImage: null, loading: 0,
};
let noticeTimer = 0;

function h(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}
function title(key) { return String(key).replaceAll("_", " ").replaceAll("-", " "); }
function white(value) { return value == null ? "—" : value === true ? "Yes" : value === false ? "No" : String(value); }
function pretty(value) { return JSON.stringify(value, null, 2); }
function el(selector) { return document.querySelector(selector); }
function find(id) { return state.photos.find(item => item.id === id); }
function current() { return find(state.focus); }
function definition() { return state.tools.find(item => item.id === state.tool); }
function report() { return state.readings; }
function apiBase() { return "/api/stations/" + state.station.id; }
function photoPath() { return "/api/photos/" + state.focus; }

async function rpc(url, method = "GET", body = null) {
  const config = {method, headers: {}};
  if (body instanceof FormData) config.body = body;
  else if (body !== null) {
    config.headers["Content-Type"] = "application/json";
    config.body = JSON.stringify(body);
  }
  const response = await fetch(url, config);
  if (!response.headers.get("content-type")?.includes("json")) {
    throw Error(response.status + " · No observation data returned.");
  }
  const data = await response.json();
  if (!response.ok) throw Error(data.error || "The operation could not be completed.");
  return data;
}

function announce(value, bad = false) {
  clearTimeout(noticeTimer);
  if (!value) { notice.hidden = true; return; }
  notice.className = bad ? "bad" : "";
  notice.textContent = value;
  notice.hidden = false;
  noticeTimer = setTimeout(() => { notice.hidden = true; }, bad ? 10000 : 4600);
}

function button(label, action, value = "", style = "secondary") {
  return '<button type="button" class="' + h(style) + '" data-act="' + h(action) +
    '" data-key="' + h(value) + '">' + h(label) + '</button>';
}

function anchor(label, href, style = "download compact", download = false) {
  return '<a href="' + h(href) + '" class="' + h(style) + '"' +
    (download ? " download" : " target=\"_blank\" rel=\"noopener\"") + ">" + h(label) + "</a>";
}

function numberField(label, name, value, step = "any", attrs = "") {
  return '<label>' + h(label) + '<input name="' + h(name) + '" type="number" value="' +
    h(value) + '" step="' + h(step) + '" ' + attrs + '></label>';
}

function field(label, name, value = "", attrs = "") {
  return '<label>' + h(label) + '<input name="' + h(name) + '" value="' + h(value) + '" ' + attrs + '></label>';
}

function area(label, name, value = "", rows = 3) {
  return '<label>' + h(label) + '<textarea name="' + h(name) + '" rows="' + rows + '">' + h(value) + '</textarea></label>';
}

function drop(label, name, pairs, selected, attributes = "") {
  return '<label>' + h(label) + '<select name="' + h(name) + '" ' + attributes + '>' +
    pairs.map(([value, text]) => '<option value="' + h(value) + '"' +
      (String(value) === String(selected) ? " selected" : "") + ">" + h(text) + "</option>").join("") + '</select></label>';
}

function checked(label, name, value) {
  return '<label class="check"><input type="checkbox" name="' + h(name) + '"' +
    (value ? " checked" : "") + ">" + h(label) + "</label>";
}

function jsonArea(label, key, data, rows = 4) { return area(label, key, pretty(data), rows); }
function readJson(fields, name) {
  try { return JSON.parse(fields.get(name)); }
  catch { throw Error("Check the coordinates or JSON in " + title(name) + "."); }
}
function asNumber(fields, name) {
  if (fields.get(name) == null || String(fields.get(name)).trim() === "") throw Error("Enter " + title(name) + ".");
  const value = Number(fields.get(name));
  if (!Number.isFinite(value)) throw Error("Use a finite number for " + title(name) + ".");
  return value;
}

function assetSrc(asset) {
  if (!asset) return "";
  return asset.url || (asset.data ? "data:" + asset.mime + ";base64," + asset.data : "");
}

function modal(name, body) {
  dialog.innerHTML = '<header class="dialog-top"><h2 id="dialog-title">' + h(name) + '</h2>' +
    button("Close", "close", "", "plain compact") + '</header><div class="dialog-body">' + body + "</div>";
  if (!dialog.open) dialog.showModal();
}

function form(name, fields, submit, {disabled = false} = {}) {
  return '<form class="form" data-form="' + h(name) + '">' + fields + '<button type="submit" class="primary"' +
    (disabled ? " disabled" : "") + ">" + h(submit) + "</button></form>";
}

function table(rows, maximum = 40) {
  if (!rows?.length) return '<p class="quiet">No readings in this view.</p>';
  const subset = rows.slice(0, maximum);
  const keys = [...new Set(subset.flatMap(Object.keys))].slice(0, 14);
  const shell = keys.map(key => '<th>' + h(title(key)) + "</th>").join("");
  const body = subset.map(row => '<tr>' + keys.map(key => {
    const value = row[key];
    return "<td>" + h(value && typeof value === "object" ? Array.isArray(value) ? value.join(", ") : JSON.stringify(value) : white(value)) + "</td>";
  }).join("") + "</tr>").join("");
  return '<div class="table-holder"><table><thead><tr>' + shell + "</tr></thead><tbody>" + body + "</tbody></table></div>";
}

function fold(label, data, isTable = false) {
  return '<details><summary>' + h(label) + "</summary>" + (isTable ? table(data) : "<pre>" + h(pretty(data)) + "</pre>") + "</details>";
}

function metrics(data) {
  const show = Object.entries(data || {}).filter(([, v]) => v === null || typeof v !== "object" || Array.isArray(v) && v.length <= 3 && v.every(
    item => item == null || typeof item !== "object"));
  return '<div class="measurements">' + show.slice(0, 10).map(([key, value]) => '<div class="datum"><span>' +
    h(title(key)) + "</span><strong>" + h(Array.isArray(value) ? value.map(white).join(" – ") : white(value)) +
    '</strong></div>').join("") + '</div>';
}

async function home() {
  const request = ++state.loading;
  const [directory, examples, catalog, budget] = await Promise.all([
    rpc("/api/stations"), rpc("/api/examples"), rpc("/api/tools"), rpc("/api/budget")
  ]);
  if (request !== state.loading) return;
  state.station = null;
  state.library = directory.stations;
  state.examples = examples.examples;
  state.tools = catalog.tools;
  state.budget = budget;
  render();
}

function libraryView() {
  const examples = state.examples.map(item => '<button type="button" class="case" data-act="sample" data-key="' + h(item.key) +
    '"><img src="' + h(item.preview) + '" alt=""><span><strong>' + h(item.title) + '</strong><span class="case-meta">' +
    h(item.description) + "</span></span></button>").join("");
  const stations = state.library.map(item => '<button type="button" class="case" data-act="station" data-key="' + h(item.id) +
    '"><span><strong>' + h(item.name) + '</strong><span class="case-meta">' + h(item.accession || "Field station") +
    " · " + item.observations + " captures · " + item.reports + " retained</span></span></button>").join("");
  return '<div class="home"><section class="hero"><div class="hero-copy"><p class="hero-overline">Fisheye field observation</p>' +
    '<h1>Read the sky<br>through the canopy.</h1><p>Locate what the camera saw. Measure sky openings, follow light paths, and keep the original field evidence beside every result.</p>' +
    '<div class="row">' + button("Open training station", "sample", "ash-lane", "primary") +
    button("Import your own view", "new", "", "plain") + '</div></div><div class="hero-plot" aria-hidden="true"><span class="plot-north">N</span>' +
    '<img src="/api/examples/ash-lane/preview" alt=""></div></section>' +
    '<div class="section-head"><h2>Field stations</h2></div>' +
    (stations ? '<div class="home-grid">' + stations + '</div>' : '<p class="empty-home">No station has been saved yet.</p>') +
    '<div class="section-head"><h2>Training stations</h2><p>Synthetic observations with known lens placement.</p></div>' +
    '<div class="home-grid">' + examples + "</div></div>";
}

function cleanResult() {
  state.readings = null;
  state.activeAsset = null;
  state.image = null;
  state.probe = null;
  state.toolImage = null;
  state.asked = null;
  state.sourceIndex = 0;
}

async function station(id, keep = false) {
  const request = ++state.loading;
  const data = await rpc("/api/stations/" + id);
  if (request !== state.loading) return;
  if (!keep || id !== state.station?.id) {
    Object.assign(state, {focus: data.photos[0]?.id || null, selected: data.photos.slice(0, 1).map(photo => photo.id),
      tab: "inspect", tool: "open-share", settings: {}, orderSettings: null, prepared: null, editor: null,
      lensPane: null, lensDraft: null, colorDraft: null, chosenColor: null, folio: null});
    cleanResult();
  }
  Object.assign(state, data);
  render();
}

function stationHeading() {
  const title = state.station;
  return '<div class="station-title"><div><div class="left-meta"><small>' + h(title.accession || "Field station") + "</small>" +
    (title.synthetic ? '<span class="tag">Synthetic</span>' : "") + '</div><h1>' + h(title.name) + '</h1></div><div class="toolbar">' +
    button("Station details", "station-edit", "", "plain") + button("Add observation", "upload", "", "primary") + "</div></div>";
}

function captureRail() {
  const content = state.photos.map(photo => '<div class="capture"><input type="checkbox" data-check="' + h(photo.id) +
    '" aria-label="Use ' + h(photo.capture.title) + '"' + (state.selected.includes(photo.id) ? " checked" : "") + '>' +
    '<button type="button" class="capture-choice" data-act="focus" data-key="' + h(photo.id) + '" aria-pressed="' +
    (state.focus === photo.id) + '"><img src="' + h(photo.preview) + '" alt=""><span>' +
    h(photo.capture.title) + '</span></button></div>').join("");
  return '<aside class="rail" aria-label="Observations"><h2>Observations</h2><p>Tick sources in comparison order.</p>' +
    content + (current() ? '<div class="h-divider"></div><div class="row">' + anchor("Original", current().original, "download compact", true) +
    button("Camera details", "camera-edit", "", "plain compact") + '</div>' : '<p class="empty">Import an overhead image to begin.</p>') +
    '</aside>';
}

const ANGULAR_RESULTS = new Set(["open-class-map.png","lens-support.png","sky-windows.png","weighted-sky.png",
  "camera-luminance.png","sky-basis.png","observed-change.png","brightness-difference.png",
  "combined-sky-map.png","source-provenance.png","conflict-review.png","combined-photo.png"]);
const NATIVE_RESULTS = new Set(["camera-display.png","camera-clipping.png","line-on-source.png","lens-on-photo.png",
  "class-overlay.png","class-margin.png","path-on-source.png","angular-transect.png","field-on-camera.png",
  "native-detail.png","validation-patches.png","trial-colors.png","lens-on-source.png","landmark-residuals.png"]);

function reportAssets() {
  const result = report();
  if (!result) return [];
  if (result.assets) return result.assets;
  return result.files.map(asset => ({...asset, label: result.asset_labels?.[asset.name] || title(asset.name)}));
}

function stageImage() {
  const active = current();
  if (state.editor && active) return {src: active.preview, label: "Native camera image", mode: "draw"};
  if (state.image?.source) return {src: state.image.source.preview, label: state.image.source.title, mode: "source"};
  if (state.prepared) {
    return {src: assetSrc(state.prepared), label: state.prepared.label, mode: NATIVE_RESULTS.has(state.prepared.name) ? "source" : ""};
  }
  if (state.activeAsset) {
    const asset = reportAssets().find(item => item.name === state.activeAsset);
    if (asset) {
      const mode = ANGULAR_RESULTS.has(asset.name) ? "angular"
        : ["direction-view.png", "compass-view.png"].includes(asset.name) ? "perspective"
        : NATIVE_RESULTS.has(asset.name) && report()?.sources.length === 1 ? "native-report" : "";
      return {src: assetSrc(asset), label: asset.label || title(asset.name), mode};
    }
  }
  return active ? {src: active.preview, label: "Native camera image", mode: "source"} : null;
}

function dot(x, y, style = "") {
  return '<i aria-hidden="true" class="marker ' + h(style) + '" style="--x:' + (100 * x) + "%;--y:" + (100 * y) + '%"></i>';
}

function rectangle(points) {
  const [a, b] = points;
  const params = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1])];
  const css = ["--x", "--y", "--w", "--h"].map((key, i) => key + ":" + 100 * params[i] + "%").join(";");
  return '<i class="marker-box" aria-hidden="true" style="' + css + '"></i>';
}

function stage() {
  const picture = stageImage();
  const selected = reportAssets().filter(asset => asset.mime === "image/png");
  const tabs = !state.editor && !state.prepared && selected.length ?
    '<div class="asset-tabs" aria-label="Result pictures">' + selected.map(item => '<button class="seg compact" type="button" data-act="asset" data-key="' +
      h(item.name) + '" aria-pressed="' + (item.name === state.activeAsset) + '">' + h(item.label) + "</button>").join("") + "</div>" : "";
  const controls = '<div class="row">' + (current() ? button("Camera", "source", "", "plain compact") : "") +
    (report()?.sources?.length > 1 ? button("Probe source " + (state.sourceIndex + 1) + "/" + report().sources.length, "cycle-source", "", "plain compact") : "") + "</div>";
  const overlay = state.editor ? state.editor.points.map(point => dot(...point)).join("") +
    (state.editor.kind === "region" && state.editor.points.length === 2 ? rectangle(state.editor.points) : "") :
    state.probe?.visual ? dot(...state.probe.visual) : "";
  const helper = state.editor ? '<div class="stage-instruction"><span>' + h(state.editor.label) + '</span>' +
    button("Cancel", "draw-cancel", "", "plain compact") + '</div>' : "";
  let frame = '<div class="empty">Add an observation to open the image stage.</div>';
  if (picture) frame = '<div class="plate" id="plate"' + (picture.mode ? ' data-mode="' + h(picture.mode) + '"' : "") +
    '><img id="stage-image" src="' + h(picture.src) + '" alt="' + h(picture.label) + '">' + overlay + "</div>";
  return '<section class="stage-wrap" aria-label="Observation image"><div class="stage-text"><h2>' +
    h(picture?.label || "Observation image") + "</h2>" + controls + "</div>" + helper + '<div class="stage">' + frame + "</div>" + tabs + "</section>";
}

function optionField(fieldSpec) {
  const {key, kind, default: fallback, title: label, step, hint} = fieldSpec;
  const value = state.settings[key] ?? fallback;
  const note = hint ? '<small>' + h(hint) + '</small>' : "";
  let control;
  if (kind === "number") control = numberField(label, key, value, step ?? "any", "required");
  else if (kind === "checkbox") control = checked(label, key, value);
  else if (kind === "pair") control = field(label, key, JSON.stringify(value), "required");
  else if (kind === "line") control = field(label, key, JSON.stringify(value), "required") + button("Mark two endpoints", "draw-line", key, "secondary compact");
  else if (key === "conflict_choices") {
    const count = Array.isArray(value) ? value.length : 0;
    control = '<label>' + h(label) + '<input name="' + h(key) + '" value=\'' + h(JSON.stringify(value)) +
      "' readonly><small>" + (count ? count + " patch choice(s) recorded; use the patch buttons below to change them."
      : "No patch choices yet. Run the view, then choose a frame per disagreement patch.") + "</small></label>";
  } else control = jsonArea(label, key, value, 4);
  if (state.tool === "sun-path" && key === "landmarks") {
    control += '<div class="row">' + button("Import directions CSV", "path-csv", "", "secondary compact") +
      anchor("Example CSV", "/api/paths/example.csv", "download compact", true) + "</div>";
  }
  if (state.tool === "field-marks" && key === "marks") control += button("Use saved field marks", "use-marks", "", "secondary compact");
  return '<div class="setting">' + control + note + "</div>";
}

function footprintEditor() {
  const footprint = state.orderSettings?.footprint ?? {bearing: 0, sweep: 360, elevation: [0, 90]};
  return '<fieldset><legend>Direction window</legend><div class="cols">' +
    numberField("Start bearing", "bearing", footprint.bearing, .1) +
    numberField("Sweep", "sweep", footprint.sweep, .1) +
    numberField("From elevation", "bottom", footprint.elevation[0], .1) +
    numberField("To elevation", "top", footprint.elevation[1], .1) + "</div></fieldset>";
}

function inspectPane() {
  const tool = definition();
  const choices = [...new Set(state.tools.map(entry => entry.group))].map(group => '<optgroup label="' + h(group) + '">' +
    state.tools.filter(entry => entry.group === group).map(entry => '<option value="' + h(entry.id) + '"' +
      (entry.id === state.tool ? " selected" : "") + ">" + h(entry.label) + '</option>').join("") + '</optgroup>').join("");
  const needs = tool.photos === 2 && state.selected.length !== 2;
  const settings = '<label>Inspection tool<select id="tool-select">' + choices + '</select></label><p class="quiet small">' + h(tool.explain) + "</p>";
  const criteria = (tool.footprint ? footprintEditor() : "") + tool.settings.map(optionField).join("");
  return '<div class="settings">' + settings + (needs ? '<p class="quiet">Tick two source observations to compare.</p>' : "") +
    form("inspection", criteria, state.busy ? "Working…" : "Run inspection", {disabled: state.busy || !current() || needs}) +
    "</div>" + resultHTML() + conflictHTML() + probeHTML();
}

function conflictHTML() {
  const value = report();
  if (!value || value.tool !== "combined-view" || !value.readings?.length) return "";
  const locked = Boolean(value.id);
  const first = value.metrics.first_capture_id;
  const second = value.metrics.second_capture_id;
  const current = new Set((value.order.settings.conflict_choices || []).map(row => row.patch + ":" + row.source));
  const rows = value.readings.slice(0, 60).map(row => {
    const firstKey = row.patch + ":" + first;
    const secondKey = row.patch + ":" + second;
    const chooseFirst = current.has(firstKey) || (!current.has(secondKey) && row.chosen_source === "first");
    const chooseSecond = current.has(secondKey) || (!current.has(firstKey) && row.chosen_source === "second");
    const frameButtons = locked
      ? '<span class="pill">' + (chooseFirst ? "Adopted Frame 1" : "Adopted Frame 2") + "</span>"
      : button("Frame 1", "conflict-pick", row.patch + ":" + first, chooseFirst ? "primary compact" : "secondary compact") +
        button("Frame 2", "conflict-pick", row.patch + ":" + second, chooseSecond ? "primary compact" : "secondary compact");
    return '<div class="conflict-row"><strong>Patch ' + h(row.patch) + ' · ' + h(row.cells) + ' cells</strong>' +
      '<small>' + h(row.center_bearing) + '°, ' + h(row.center_elevation) + '° · ' +
      h(row.first_capture_class) + ' vs ' + h(row.second_capture_class) + '</small><div class="row">' +
      frameButtons + '</div></div>';
  }).join("");
  const unresolved = value.readings.filter(row =>
    !(value.order.settings.conflict_choices || []).some(choice => choice.patch === row.patch)).length;
  const intro = locked
    ? "These capture choices are fixed in the retained record. Click the maps to read each adopted original."
    : "Both frames saw these directions differently (wind movement). Pick which frame supplies each patch; the combined sky share recalculates when you rerun.";
  return '<section class="source-evidence"><h3>Disagreement patches</h3>' +
    '<p class="small">' + h(intro) + '</p>' +
    (!locked && unresolved ? '<p class="warn">' + unresolved + ' patch(es) still use Frame 1 by default.</p>' :
      '<p class="tag">Every patch has an explicit choice.</p>') +
    '<div class="detail-list">' + rows + "</div></section>";
}

function resultHTML() {
  const value = report();
  if (!value) return "";
  const archive = value.id ? '<div class="row">' +
    anchor("Print sheet", "/api/retained/" + value.id + "/sheet") +
    anchor("Evidence ZIP", "/api/retained/" + value.id + "/packet", "download compact", true) +
    anchor("Results-only ZIP", "/api/retained/" + value.id + "/packet?originals=0", "download compact", true) +
    button("Review", "review", "", "plain compact") + button("Check sources", "sources", "", "plain compact") + '</div>'
    : button("Retain observation", "retain", "", "primary");
  const downloads = reportAssets().filter(asset => asset.mime.includes("csv")).map(asset => anchor(
    asset.label, assetSrc(asset), "download compact", true)).join("");
  const warnings = (value.warnings || []).map(row => '<p class="warn">' + h(row) + '</p>').join("");
  return '<section class="result-body"><h3>' + h(value.name || value.title) + '</h3><p class="small">' + h(value.conclusion) +
    "</p>" + metrics(value.metrics) + archive +
    (downloads ? '<div class="row">' + downloads + "</div>" : "") +
    fold("All measurements", value.metrics) + fold("Readings · " + value.row_count, value.readings, true) +
    fold("Source observations", value.sources) + (value.reviews?.length ? fold("Review history", value.reviews, true) : "") + warnings + "</section>";
}

function probeHTML() {
  const data = state.probe;
  if (!data) return "";
  const reading = data.results.readings[0];
  const witness = data.results.witness ? '<img class="image-data" alt="Magnified source pixels" src="' + h(assetSrc(data.results.witness)) + '">' : "";
  const note = data.results.provenance_note ? '<p class="warn">' + h(data.results.provenance_note) +
    (data.results.other_source_index !== undefined ?
      ' ' + button("Inspect the opposite capture", "cycle-source", "", "secondary compact") : "") + "</p>" : "";
  return '<section class="source-evidence"><h3>Point evidence</h3>' + note + metrics({
    class: reading.class, direction: reading.direction, camera_pixel: reading.pixel || reading.camera_px,
    rgb: reading.rgb, provenance: data.results.provenance
  }) + witness + fold("Exact point readout", reading) + "</section>";
}

function footerHelp() {
  return current() ? '<p class="lab-note">Click an image or direction map to read its source point.</p>' : "";
}

function preparation() {
  if (!current()) return '<p class="quiet">Add an observation to prepare its field geometry.</p>';
  const mode = state.lensPane?.mode || "lens";
  const options = '<div class="row">' + [["lens","Lens"],["colors","Classify"],["marks","Field marks"]].map(([key, label]) =>
    button(label, "prepare-view", key, mode === key ? "primary compact" : "secondary compact")).join("") + "</div>";
  let body;
  if (mode === "colors") body = classifierPanel();
  else if (mode === "marks") body = fieldPanel();
  else body = lensPanel();
  return options + '<div class="h-divider"></div>' + body + probeHTML();
}

function defaultLens() {
  return {center_uv: [.5, .5], radius_over_short_side: .49, maximum_zenith: 90, north_clockwise: 0, model: "equidistant"};
}

async function choosePrepare(mode, keep = false) {
  if (!current()) return;
  const selected = state.focus;
  state.tab = "prepare";
  state.lensPane = null;
  state.probe = null;
  state.image = null;
  state.activeAsset = null;
  state.editor = null;
  if (!keep) state.prepared = null;
  if (mode === "lens") {
    const revisions = await rpc(photoPath() + "/lens");
    if (state.focus !== selected) return;
    state.lensPane = {mode: "lens", versions: revisions.versions, active: revisions.active};
    const initial = revisions.versions.find(item => item.id === revisions.active)?.lens || defaultLens();
    state.lensDraft = {lens: {...initial}, title: "Field geometry " + (revisions.versions.length + 1), marks: []};
    state.chosenLens = null;
  } else if (mode === "colors") {
    if (!keep || !state.colorDraft) state.colorDraft = structuredClone(current().classification);
    state.chosenColor = null;
    state.lensPane = {mode};
  } else if (mode === "marks") {
    const result = await rpc(photoPath() + "/marks");
    if (state.focus !== selected) return;
    state.marks = result.marks;
    state.lensPane = {mode};
  }
  render();
}

function lensPanel() {
  const lens = state.lensDraft?.lens || defaultLens();
  const draft = state.lensDraft || {title: "Field geometry", marks: []};
  const models = [["equidistant", "Equidistant"], ["equisolid", "Equisolid"], ["orthographic", "Orthographic"], ["stereographic", "Stereographic"]];
  const settings = field("Version title", "title", draft.title, "required maxlength=78") +
    '<div class="cols">' + numberField("Center X", "center_x", lens.center_uv[0], "any", "min=0 max=1 required") +
    numberField("Center Y", "center_y", lens.center_uv[1], "any", "min=0 max=1 required") + '</div>' +
    button("Pick center on photo", "draw-center", "", "secondary compact") +
    '<div class="cols">' + numberField("Radius / short side", "radius", lens.radius_over_short_side, "any", "min=.065 max=2.5 required") +
    numberField("Max angle from zenith", "zenith", lens.maximum_zenith, "any", "min=10 max=90 required") +
    numberField("North rotation (°)", "north", lens.north_clockwise, "any", "min=-720 max=720 required") +
    drop("Lens law", "model", models, lens.model) +
    "</div><details><summary>Optional surveyed landmarks</summary>" +
    jsonArea("Image position and known bearing/elevation", "marks", draft.marks, 6) +
    '<div class="row">' + button("Pick a landmark", "draw-landmark", "", "secondary compact") + "</div></details>";
  const preview = state.chosenLens ? '<div class="source-evidence">' + metrics(state.chosenLens.fit.metrics) +
    fold("Lens geometry and support", state.chosenLens.coverage) +
    (state.chosenLens.fit.residuals ? fold("Landmark residuals", state.chosenLens.fit.residuals, true) : "") +
    '<div class="row">' + state.chosenLens.assets.map(asset => button(asset.label, "lens-picture", asset.name, "secondary compact")).join("") +
    '</div>' + button("Retain reviewed lens", "save-lens", "", "primary") + "</div>" : "";
  const versions = (state.lensPane?.versions || []).map(record => '<div class="revision"><strong>' + h(record.title) + '</strong><small>Version ' +
    record.ordinal + " · " + h(record.lens.model) + '</small>' +
    (record.id === state.lensPane.active ? '<span class="tag">Active lens</span>' : button("Activate version", "activate-lens", record.id, "secondary compact")) +
    "</div>").join("");
  return '<h2>Lens placement</h2><p class="small">Use a level lens circle and a known compass direction. Preview the geometry before keeping a version.</p>' +
    form("lens-preview", settings, "Preview lens") + preview +
    (versions ? '<div class="h-divider"></div><h3>Lens history</h3><div class="detail-list">' + versions + "</div>" : "");
}

function classifierPanel() {
  const options = state.colorDraft ?? current().classification;
  const controls = '<div class="cols">' + numberField("Ambiguity margin", "margin", options.margin, "any", "min=.01 max=.6 required") +
    numberField("Color limit", "distance", options.distance_ceiling, "any", "min=.1 max=1.2 required") +
    numberField("Clipping guard", "clipping", options.clipping_guard, "any", "min=.85 max=1 required") + "</div>" +
    '<div class="row">' + button("Mark sky", "draw-region", "sky", "secondary compact") +
    button("Mark obstruction", "draw-region", "obstruction", "secondary compact") +
    button("Exclude region", "draw-region", "exclude", "secondary compact") + "</div>" +
    "<details><summary>Source polygons</summary>" + jsonArea("Color samples", "regions", options.regions, 7) + "</details>";
  const prepared = state.chosenColor ? '<div class="source-evidence">' + fold("Sample support", state.chosenColor.classifier.samples, true) +
    button("Retain reviewed colors", "save-color", "", "primary") + "</div>" : "";
  return '<h2>Sky color evidence</h2><p class="small">Mark a few known sky and obstruction patches on the original to check borderline colors.</p>' +
    form("color-preview", controls, "Preview classification") + prepared;
}

function fieldPanel() {
  return '<h2>Field landmarks</h2><p class="small">Keep notes at a camera point or a surveyed direction.</p><div class="row">' +
    button("Pick a native mark", "draw-mark", "", "secondary compact") + button("Add direction note", "direction-mark", "", "secondary compact") +
    button("Import field log", "notes-import", "", "secondary compact") + "</div><div class='row'>" +
    anchor("Notes JSON", photoPath() + "/mark-log/json", "download compact", true) +
    anchor("Notes CSV", photoPath() + "/mark-log/csv", "download compact", true) + "</div>" +
    '<div class="detail-list">' + state.marks.map(mark => '<div class="mark-row"><strong>' + h(mark.title) + "</strong><small>" +
      h(title(mark.role)) + " · " + h(mark.native?.join(", ") || mark.direction?.join(", ") || "") + '</small>' +
      (mark.note ? '<p class="small">' + h(mark.note) + '</p>' : "") + button("Read marked place", "read-mark", mark.id, "secondary compact") +
      "</div>").join("") + "</div>";
}

function recordsPane() {
  let toolbar = "";
  if (state.reports.length) toolbar = '<div class="row">' + button("Select field folio", "folio", "", "secondary compact") + "</div>";
  const options = state.reports.map(row => '<div class="record-row">' +
    button(row.name, "record", row.id, "plain compact") + '<small>' + h(title(row.tool)) + " · " + h(row.created_at.replace("T", " ").slice(0, 16)) +
    "</small></div>").join("");
  return '<h2>Station records</h2>' + toolbar +
    (options ? '<div class="detail-list">' + options + '</div>' : '<p class="quiet small">Run an inspection and retain it to have its numbers and source basis here.</p>') +
    resultHTML() + conflictHTML() + probeHTML();
}

function panel() {
  const tabLabels = [["inspect", "Inspect"], ["prepare", "Prepare"], ["records", "Records"]];
  const tabbar = tabLabels.map(([key, label]) => '<button type="button" data-act="tab" data-key="' + key +
    '" aria-current="' + (state.tab === key) + '">' + h(label) + "</button>").join("");
  const body = state.tab === "inspect" ? inspectPane() : state.tab === "prepare" ? preparation() : recordsPane();
  return '<aside class="panel" aria-label="Work controls"><nav class="tabs">' + tabbar + '</nav><div class="panel-area">' +
    body + footerHelp() + "</div></aside>";
}

function render() {
  main.innerHTML = state.station && definition() ? stationHeading() + '<div class="bench">' +
    captureRail() + stage() + panel() + "</div>" : libraryView();
}

function stationDialog(existing = false) {
  const old = existing ? state.station : {};
  modal(existing ? "Station details" : "New field station", form(existing ? "station-edit" : "station-new",
    field("Station name", "name", old.name, "required maxlength=96") +
    field("Field reference", "accession", old.accession, "maxlength=70") + area("Field note", "note", old.note, 3) +
    checked("Synthetic observation", "synthetic", old.synthetic), existing ? "Save station details" : "Create station"));
}

function fileHelp() {
  return '<p class="quiet small">Import PNG, JPEG, WebP or single-frame TIFF up to ' +
    Math.round(state.budget.camera_file_bytes / (1024 * 1024)) + " MiB. Sticker cutouts and transparent image areas stay unmeasured.</p>";
}

function captureDialog(edit = false) {
  const old = edit ? current().capture : {recorded_at: new Date().toISOString().split(".")[0] + "+00:00", exposure_role: "survey"};
  if (!state.station) return stationDialog();
  const selector = edit ? "" : '<label>Camera image<input type="file" name="image" accept="image/png,image/jpeg,image/webp,image/tiff" required></label>' + fileHelp();
  modal(edit ? "Camera observation details" : "Add camera observation",
    form(edit ? "camera-edit" : "camera-upload", selector + field("Capture title", "title", old.title, "maxlength=96") +
      field("Recorded at (ISO, with time zone)", "recorded_at", old.recorded_at, "required") +
      drop("Capture purpose", "exposure_role", [["survey", "Survey"], ["reference", "Reference"], ["comparison","Comparison"]], old.exposure_role) +
      field("Timing label", "time_label", old.time_label, "maxlength=70") + area("Camera note", "note", old.note, 3),
      edit ? "Save capture details" : "Import observation"));
}

function rememberInspection() {
  const formNode = el('form[data-form="inspection"]');
  if (!formNode) return;
  const fields = new FormData(formNode);
  const settings = {};
  for (const option of definition().settings) {
    if (option.kind === "checkbox") settings[option.key] = fields.has(option.key);
    else if (option.kind === "number") settings[option.key] = asNumber(fields, option.key);
    else if (option.key === "conflict_choices") settings[option.key] = state.settings.conflict_choices ?? option.default;
    else settings[option.key] = readJson(fields, option.key);
  }
  const footprint = definition().footprint ? {
    bearing: asNumber(fields, "bearing"), sweep: asNumber(fields, "sweep"),
    elevation: [asNumber(fields, "bottom"), asNumber(fields, "top")]
  } : null;
  state.settings = settings;
  state.orderSettings = {footprint};
}

function currentOrder() {
  const tool = definition();
  const sources = tool.photos === 2 ? state.selected : [state.selected.length === 1 ? state.selected[0] : state.focus];
  return {photos: sources, settings: state.settings, ...(state.orderSettings?.footprint ? {footprint: state.orderSettings.footprint} : {})};
}

async function run() {
  rememberInspection();
  if (state.busy || !state.focus) return;
  const expected = state.station.id + ":" + state.tool;
  state.busy = true;
  state.probe = null;
  render();
  try {
    const value = await rpc(apiBase() + "/tools/" + state.tool, "POST", currentOrder());
    if (expected !== state.station?.id + ":" + state.tool) return;
    displayReport(value);
    announce("Inspection ready.");
  } finally {
    state.busy = false;
    render();
  }
}

function displayReport(value) {
  state.readings = value;
  state.activeAsset = reportAssets().find(entry => entry.mime === "image/png")?.name || null;
  state.prepared = null;
  state.image = null;
  state.probe = null;
  state.sourceIndex = 0;
}

function lensInput() {
  const candidate = el('form[data-form="lens-preview"]');
  if (!candidate) return state.lensDraft;
  const fd = new FormData(candidate);
  return {
    title: fd.get("title"),
    lens: {center_uv: [asNumber(fd, "center_x"), asNumber(fd, "center_y")],
      radius_over_short_side: asNumber(fd, "radius"), maximum_zenith: asNumber(fd, "zenith"),
      north_clockwise: asNumber(fd, "north"), model: fd.get("model")},
    marks: readJson(fd, "marks"),
  };
}

function colorInput() {
  const formNode = el('form[data-form="color-preview"]');
  if (!formNode) return state.colorDraft;
  const fd = new FormData(formNode);
  return {margin: asNumber(fd, "margin"), distance_ceiling: asNumber(fd, "distance"), clipping_guard: asNumber(fd, "clipping"),
    regions: readJson(fd, "regions")};
}

async function lensTest() {
  state.lensDraft = lensInput();
  const response = await rpc(photoPath() + "/lens", "POST", {...state.lensDraft, revision: current().revision});
  state.chosenLens = response;
  state.prepared = response.assets[0];
  state.probe = null;
  render();
  announce("Lens preview ready.");
}

async function saveLens() {
  if (!state.chosenLens) return;
  const response = await rpc(photoPath() + "/lens", "POST", {...lensInput(), revision: current().revision,
    expected: state.chosenLens.fingerprint, retain: true});
  state.prepared = response.assets[0];
  await station(state.station.id, true);
  await choosePrepare("lens", true);
  announce("Lens retained.");
}

async function classifyTest() {
  state.colorDraft = colorInput();
  const data = await rpc(photoPath() + "/classify", "POST", {revision: current().revision, classifier: state.colorDraft});
  state.chosenColor = data;
  state.prepared = data.preview;
  state.probe = null;
  render();
  announce("Classification preview ready.");
}

async function saveColors() {
  if (!state.chosenColor) return;
  const response = await rpc(photoPath() + "/classify", "POST", {revision: current().revision, classifier: colorInput(),
    expected: state.chosenColor.fingerprint, retain: true});
  state.prepared = response.preview;
  state.colorDraft = response.photo.classification;
  await station(state.station.id, true);
  await choosePrepare("colors", true);
  announce("Color samples retained.");
}

function enterDraw(kind, label, key = "") {
  state.probe = null;
  state.editor = {kind, key, label, points: []};
  render();
}

function handleDraw(uv) {
  const edit = state.editor;
  if (!edit) return;
  edit.points.push(uv);
  if (["line", "region"].includes(edit.kind) && edit.points.length === 1) {render(); return;}
  state.editor = null;
  if (edit.kind === "center") {
    state.lensDraft.lens.center_uv = uv;
    state.chosenLens = null;
    state.prepared = null;
  } else if (edit.kind === "line") {
    state.settings[edit.key] = edit.points;
    cleanResult();
  } else if (edit.kind === "region") {
    const [a, b] = edit.points;
    if (Math.abs(a[0] - b[0]) < .015 || Math.abs(a[1] - b[1]) < .015) throw Error("Pick opposite corners with room between them.");
    const name = title(edit.key) + " " + (state.colorDraft.regions.length + 1);
    state.colorDraft.regions.push({kind: edit.key, label: name,
      vertices: [a, [b[0], a[1]], b, [a[0], b[1]]]});
    state.chosenColor = null;
    state.prepared = null;
  }
  render();
  if (edit.kind === "landmark") landmarkDialog(uv);
  if (edit.kind === "mark") markDialog(uv);
}

function landmarkDialog(uv) {
  state.pendingPoint = uv;
  modal("Surveyed landmark",
    form("survey-landmark", field("Landmark label", "title", "", "required maxlength=65") +
      '<div class="cols">' + numberField("Bearing (°)", "bearing", 0, .1, "required") +
      numberField("Elevation (°)", "elevation", 35, .1, "min=0 max=90 required") + "</div>" +
      numberField("Mark confidence", "confidence", 1, .05, "min=.05 max=1 required"),
      "Add landmark"));
}

function markDialog(uv = null) {
  state.pendingPoint = uv;
  modal("Field mark", form("field-mark", field("Mark title", "title", "", "required maxlength=80") +
    (uv ? "" : '<div class="cols">' + numberField("Bearing", "bearing", 0, .1, "required") +
      numberField("Elevation", "elevation", 40, .1, "min=0 max=90 required") + "</div>") +
    drop("Mark type", "role", [["reference","Reference"],["opening","Opening"],["obstruction","Obstruction"],["note","Note"]], "reference") +
    area("Field note", "note", "", 3), "Keep field mark"));
}

function directionFromAngular(uv, footprint) {
  const x = (uv[0] * 960 - 69) / 835;
  const y = (uv[1] * 490 - 57) / 352;
  if (x < 0 || x > 1 || y < 0 || y > 1) return null;
  const [low, high] = footprint.elevation.map(value => Math.sin(value * Math.PI / 180));
  const bearing = (footprint.bearing + x * footprint.sweep) % 360;
  const elevation = Math.asin(low + (high - low) * (1 - y)) * 180 / Math.PI;
  return [bearing, elevation];
}

async function probeRequest(point, space, context, screen = null) {
  const activeReport = report();
  const archived = activeReport?.id;
  let uri = photoPath() + "/probe";
  const body = {space, positions: [point], radius: 7, witness: true};
  if (context === "retained" && archived && activeReport.tool === "combined-view") {
    uri = "/api/retained/" + activeReport.id + "/combined-probe";
  } else if (context === "retained" && archived) {
    uri = "/api/retained/" + activeReport.id + "/probe";
    body.source_index = state.sourceIndex;
  } else if (context === "retained" && activeReport?.sources?.length) {
    const source = activeReport.sources[state.sourceIndex];
    uri = "/api/photos/" + source.id + "/probe";
    body.lens = source.lens_id;
  }
  const results = await rpc(uri, "POST", body);
  if (results.adopted_source_index !== undefined) state.sourceIndex = results.adopted_source_index;
  state.probe = {results, visual: screen};
  render();
}

async function plateEvent(event) {
  const image = event.target.closest("#stage-image");
  if (!image) return false;
  const plate = image.closest("#plate");
  if (!plate.dataset.mode) return false;
  const loc = image.getBoundingClientRect();
  const x = (event.clientX - loc.left) / loc.width;
  const y = (event.clientY - loc.top) / loc.height;
  if (x < 0 || x > 1 || y < 0 || y > 1) return false;
  const uv = [Number(x.toFixed(6)), Number(y.toFixed(6))];
  if (state.editor) {
    handleDraw(uv);
    return true;
  }
  const active = report();
  const mode = plate.dataset.mode;
  if (mode === "perspective") {
    const uri = apiBase() + "/perspective-probe";
    const args = active.id ? {record: active.id, position: [uv]} :
      {photo: active.sources[0].id, view: active.order.settings, position: [uv]};
    const readings = await rpc(uri, "POST", args);
    state.probe = {results: readings, visual: uv};
    render();
    return true;
  }
  if (mode === "angular") {
    const position = directionFromAngular(uv, active.order.footprint);
    if (position) await probeRequest(position, "direction", "retained", uv);
    return true;
  }
  const context = mode === "native-report" || state.image?.source ? "retained" : "live";
  await probeRequest(uv, "image", context, uv);
  return true;
}

async function focus(id) {
  if (!find(id)) return;
  state.focus = id;
  if (state.selected.length < 2) state.selected = [id];
  state.editor = null;
  state.prepared = null;
  state.chosenColor = null;
  state.activeAsset = null;
  state.image = null;
  state.probe = null;
  if (state.tab === "prepare") await choosePrepare(state.lensPane?.mode || "lens");
  else render();
}

async function openRecord(id) {
  const data = await rpc("/api/retained/" + id);
  displayReport(data);
  state.tab = "records";
  render();
}

function folioDialog() {
  state.folio = null;
  const options = state.reports.slice(0, 16).map(row => '<label class="check"><input type="checkbox" name="report" value="' +
    h(row.id) + '" checked>' + h(row.name) + '</label>').join("");
  modal("Prepare a station folio",
    form("folio-preview", field("Folio title", "title", state.station.name, "maxlength=80") + options, "Preview field folio"));
}

function folioResult(doc, title, ids) {
  const query = new URLSearchParams({title});
  for (const id of ids) query.append("report", id);
  const url = apiBase() + "/folio/";
  modal(doc.title, metrics({report_count: doc.report_count, source_count: doc.source_count,
    needs_review: doc.needs_review, lens_versions_to_recheck: doc.lens_versions_to_recheck}) +
    '<div class="row">' + anchor("Print folio", url + "print.html?" + query) +
    anchor("Index CSV", url + "index.csv?" + query, "download compact", true) + "</div>" +
    '<img class="image-data" src="' + h(url + "contact.png?" + query) + '" alt="Inspection contact sheet">' +
    fold("Folio readings", doc.readings, true));
}

function packetDialog() {
  modal("Open an evidence packet",
    form("packet-view", '<label>CrownWindow ZIP<input type="file" name="packet" accept=".zip,application/zip" required></label>', "Read verified snapshot"));
}

function packetResult(packet) {
  const assets = packet.figures.map(figure => '<figure><img src="data:image/png;base64,' + h(figure.data) +
    '" alt="' + h(figure.label) + '"><figcaption>' + h(figure.label) + "</figcaption></figure>").join("");
  modal(packet.name, "<p>" + h(packet.conclusion) + "</p>" +
    '<p class="tag">Fingerprint verified</p>' + metrics(packet.metrics) +
    '<div class="gallery">' + assets + "</div>" + fold("Original camera evidence", packet.sources, true) +
    fold("Archive integrity", packet.receipt.files, true) + fold("Stored readings", packet.readings, true) +
    button("Download verification receipt", "packet-receipt", "", "secondary"));
}

const actions = {
  home: home,
  new: () => stationDialog(),
  station: (id) => station(id),
  "station-edit": () => stationDialog(true),
  upload: () => captureDialog(false),
  "camera-edit": () => captureDialog(true),
  sample: async (key) => { announce("Preparing the labelled synthetic observations…"); const data = await rpc("/api/examples/" + key, "POST"); await station(data.station.id); },
  focus,
  close: () => dialog.close(),
  tab: async (entry) => {
    if (entry === "prepare") return choosePrepare(state.lensPane?.mode || "lens");
    state.tab = entry;
    state.editor = null;
    state.prepared = null;
    render();
  },
  source: () => { state.activeAsset = null; state.prepared = null; state.image = null; state.editor = null; state.probe = null; render(); },
  asset: (key) => { state.activeAsset = key; state.prepared = null; state.image = null; state.probe = null; render(); },
  "cycle-source": () => {
    const sources = report()?.sources;
    if (!sources?.length) return;
    state.sourceIndex = (state.sourceIndex + 1) % sources.length;
    const active = sources[state.sourceIndex];
    state.image = {source: {...active, title: active.capture.title}};
    state.activeAsset = null;
    state.prepared = null;
    state.probe = null;
    render();
  },
  "prepare-view": (view) => choosePrepare(view),
  "conflict-pick": async (value) => {
    const [patchText, source] = value.split(":");
    const patch = Number(patchText);
    rememberInspection();
    const choices = Array.isArray(state.settings.conflict_choices) ? state.settings.conflict_choices : [];
    state.settings.conflict_choices = choices.filter(row => row.patch !== patch);
    state.settings.conflict_choices.push({patch, source});
    await run();
    announce("Patch " + patch + " assigned; the combined view was recalculated.");
  },
  "draw-line": (key) => { rememberInspection(); enterDraw("line", "Mark the two endpoints on the original camera image.", key); },
  "draw-center": () => { state.lensDraft = lensInput(); enterDraw("center", "Mark the center of the lens circle."); },
  "draw-landmark": () => { state.lensDraft = lensInput(); enterDraw("landmark", "Click the position corresponding to your surveyed landmark."); },
  "draw-region": (key) => { state.colorDraft = colorInput(); enterDraw("region", "Pick opposite corners for the " + title(key) + " sample.", key); },
  "draw-mark": () => enterDraw("mark", "Select the observed camera position."),
  "direction-mark": () => markDialog(),
  "draw-cancel": () => { state.editor = null; render(); },
  "lens-picture": (key) => { state.prepared = state.chosenLens.assets.find(asset => asset.name === key); state.probe = null; render(); },
  "save-lens": saveLens,
  "save-color": saveColors,
  "activate-lens": async (key) => {
    await rpc(photoPath() + "/active-lens", "PUT", {lens: key, revision: current().revision});
    await station(state.station.id, true);
    await choosePrepare("lens");
    announce("Lens version activated.");
  },
  "read-mark": async (key) => {
    const mark = state.marks.find(row => row.id === key);
    if (!mark) return;
    state.prepared = null;
    state.activeAsset = null;
    const space = mark.native ? "image" : "direction";
    await probeRequest(mark.native || mark.direction, space, "live", mark.native);
  },
  "use-marks": async () => {
    rememberInspection();
    const data = await rpc(photoPath() + "/marks");
    if (!data.marks.length) throw Error("This observation has no saved field marks.");
    state.settings.marks = data.marks.map(mark =>
      ({title: mark.title, native: mark.native, direction: mark.direction, role: mark.role, note: mark.note}));
    render();
  },
  "notes-import": () => modal("Import field notes", form("notes-import",
    '<label>Field notes JSON<input type="file" name="notes" accept=".json,application/json" required></label>', "Check source and notes")),
  "notes-save": async () => {
    const payload = new FormData();
    payload.append("notes", state.uploads.notes);
    payload.append("revision", current().revision);
    payload.append("stage", "retain");
    const data = await rpc(photoPath() + "/mark-log", "POST", payload);
    dialog.close();
    await choosePrepare("marks");
    announce(data.saved.length + " field marks retained.");
  },
  "path-csv": () => {
    rememberInspection();
    modal("Import direction path", form("path-csv",
      '<label>Timed direction CSV<input type="file" name="directions" accept=".csv,text/csv" required></label>', "Import the recorded directions"));
  },
  retain: () => modal("Retain observation", form("retain",
    field("Result name", "name", report().title, "required maxlength=85"), "Save observed result")),
  record: openRecord,
  review: () => modal("Review this observation", form("review",
    field("Reviewer", "name", "", "required maxlength=75") +
    drop("Decision", "status", [["observed", "Observed"], ["confirmed", "Confirmed"], ["needs_revisit", "Needs revisit"]], "observed") +
    area("Review note", "note"), "Save review")),
  sources: async () => {
    const details = await rpc("/api/retained/" + report().id + "/sources");
    const rows = details.sources.map(value =>
      ({title: value.title, source_bytes_match: value.source_bytes_match, active_lens_changed: value.active_lens_changed,
        capture_metadata_changed: value.capture_metadata_changed, current_classifier_changed: value.current_classifier_changed}));
    modal("Current and retained source evidence", table(rows) + '<div class="row">' + details.sources
      .filter(s => s.present).map(s => anchor(s.title, s.original, "download compact", true)).join("") + "</div>" + fold("Exact retained settings", details.sources));
  },
  folio: folioDialog,
  packets: packetDialog,
  "packet-receipt": async () => {
    const payload = new FormData();
    payload.append("packet", state.uploads.packet);
    const answer = await fetch("/api/packets/receipt", {method: "POST", body: payload});
    if (!answer.ok) throw Error("The verification receipt could not be generated.");
    const url = URL.createObjectURL(await answer.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = "evidence-receipt.png";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  },
};

const submissions = {
  "station-new": async (fd) => {
    const data = await rpc("/api/stations", "POST", {name: fd.get("name"), accession: fd.get("accession"),
      note: fd.get("note"), synthetic: fd.has("synthetic")});
    dialog.close();
    await station(data.station.id);
  },
  "station-edit": async (fd) => {
    await rpc(apiBase(), "PUT", {revision: state.station.revision,
      station: {name: fd.get("name"), accession: fd.get("accession"), note: fd.get("note"), synthetic: fd.has("synthetic")}});
    dialog.close();
    await station(state.station.id, true);
  },
  "camera-upload": async (fd) => {
    const payload = new FormData();
    payload.append("image", fd.get("image"));
    payload.append("capture", JSON.stringify(captureForm(fd)));
    const data = await rpc(apiBase() + "/photos", "POST", payload);
    dialog.close();
    await station(state.station.id, true);
    await focus(data.photo.id);
    announce("Camera observation imported.");
  },
  "camera-edit": async (fd) => {
    await rpc(photoPath(), "PUT", {revision: current().revision, capture: captureForm(fd)});
    dialog.close();
    await station(state.station.id, true);
  },
  "inspection": run,
  "lens-preview": lensTest,
  "color-preview": classifyTest,
  "survey-landmark": async (fd) => {
    state.lensDraft.marks.push({title: fd.get("title"), direction: [asNumber(fd, "bearing"), asNumber(fd, "elevation")],
      confidence: asNumber(fd, "confidence"), image: state.pendingPoint});
    state.chosenLens = null;
    state.prepared = null;
    dialog.close();
    render();
    el('form[data-form="lens-preview"] details').open = true;
  },
  "field-mark": async (fd) => {
    const coordinates = state.pendingPoint ? {native: state.pendingPoint} : {
      direction: [asNumber(fd, "bearing"), asNumber(fd, "elevation")]};
    await rpc(photoPath() + "/marks", "POST", {revision: current().revision,
      mark: {title: fd.get("title"), role: fd.get("role"), note: fd.get("note"), ...coordinates}});
    dialog.close();
    await choosePrepare("marks");
  },
  "notes-import": async (fd) => {
    const payload = new FormData();
    state.uploads.notes = fd.get("notes");
    payload.append("notes", state.uploads.notes);
    payload.append("revision", current().revision);
    const data = await rpc(photoPath() + "/mark-log", "POST", payload);
    modal("Check the portable field notes", metrics(data) + fold("New field marks", data.new, true) +
      (data.new.length ? button("Retain new field marks", "notes-save", "", "primary") : '<p class="quiet">No new marks for these pixels.</p>'));
  },
  "path-csv": async (fd) => {
    const data = await rpc("/api/paths/import", "POST", fd);
    state.settings.landmarks = data.landmarks;
    dialog.close();
    render();
    announce(data.count + " supplied direction landmarks imported.");
  },
  "retain": async (fd) => {
    const draft = report();
    const saved = await rpc(apiBase() + "/tools/" + draft.tool, "POST",
      {...draft.order, retain: true, name: fd.get("name"), expected: draft.fingerprint});
    dialog.close();
    await station(state.station.id, true);
    displayReport(saved);
    render();
    announce("Observation retained.");
  },
  "review": async (fd) => {
    const key = report().id;
    await rpc("/api/retained/" + key + "/reviews", "POST", {name: fd.get("name"), status: fd.get("status"),
      note: fd.get("note"), seen_reviews: report().reviews?.length || 0});
    dialog.close();
    await openRecord(key);
    announce("Review appended.");
  },
  "folio-preview": async (fd) => {
    const title = fd.get("title");
    const ids = fd.getAll("report");
    if (!ids.length) throw Error("Select at least one retained observation.");
    const doc = await rpc(apiBase() + "/folio", "POST", {title, reports: ids});
    folioResult(doc, title, ids);
  },
  "packet-view": async (fd) => {
    state.uploads.packet = fd.get("packet");
    const packet = await rpc("/api/packets/view", "POST", fd);
    packetResult(packet);
  },
};

function captureForm(fd) {
  return {title: fd.get("title"), recorded_at: fd.get("recorded_at"), note: fd.get("note"),
    time_label: fd.get("time_label"), exposure_role: fd.get("exposure_role")};
}

async function safely(action) {
  try { await action(); }
  catch (error) { announce(error.message || "The workbench could not complete the operation.", true); }
}

document.body.addEventListener("click", (event) => safely(async () => {
  if (await plateEvent(event)) return;
  const button = event.target.closest("button[data-act]");
  if (!button) return;
  const action = actions[button.dataset.act];
  if (action) await action(button.dataset.key);
}));

document.body.addEventListener("submit", (event) => {
  const form = event.target.closest("form[data-form]");
  if (!form) return;
  event.preventDefault();
  safely(() => submissions[form.dataset.form](new FormData(form)));
});

main.addEventListener("change", (event) => safely(async () => {
  const target = event.target;
  if (target.id === "tool-select") {
    state.tool = target.value;
    state.settings = {};
    state.orderSettings = null;
    state.prepared = null;
    cleanResult();
    render();
  }
  const id = target.dataset.check;
  if (!id) return;
  if (target.checked && state.selected.length >= 4) { target.checked = false; throw Error("Select at most four observations."); }
  if (state.tab === "inspect") rememberInspection();
  state.selected = target.checked ? [...state.selected, id] : state.selected.filter(i => i !== id);
  render();
}));

safely(home);
