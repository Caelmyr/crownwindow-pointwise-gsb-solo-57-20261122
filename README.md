# CrownWindow

CrownWindow is a local workbench for looking up through a tree canopy from a fixed station. It binds directional measurements to actual fisheye camera support and records what could not be seen. A camera location, a level lens model and a field bearing are the basis of each direction.

## Run

```sh
docker compose up --build -d --wait
```

Open http://127.0.0.1:19963. Stop with `docker compose down`. The default named volume keeps stations, original image bytes, lens versions and retained inspections across container replacement. Set `CROWNWINDOW_PORT` to use another loopback port. With Python 3.11 and the locked requirements installed, `./start.sh` runs the native build; `CROWNWINDOW_DATA`, `CROWNWINDOW_PYTHON` and `PORT` select its local data path and listener.

Ash Lane, Brick Courtyard and Field Edge are labelled synthetic stations. Each has an initial image and a rotated repeat with independently generated missing areas. Select **Open training station** to load two source images. To load a real survey, create a station and add an overhead photograph with the camera time and purpose.

## Make and revisit an observation

1. Under **Prepare → Lens**, check the center, radius, direction rotation and radial law. Preview before you retain a lens. If you have at least four well-spaced landmarks with bearings and elevations, mark their camera positions to fit the placement and review the residuals. Use the version list to reactivate any saved lens.
2. Under **Prepare → Classify**, mark independent sky and obstruction examples on the source; a planned exclusion can keep a lens housing or a temporary artefact outside measured support. Preview the classes and retain the specific classifier. Independent check patches in **Color model validation** do not change the training samples.
3. In **Inspect**, choose source signal, clipping, a native line or evidence windows to investigate the original. **Lens coverage** shows the field that falls inside the optics and which part actually reached the camera. **Look toward bearing** resamples in ordinary perspective and lets you inspect a particular ray.
4. **Open sky area** uses equal-solid-angle cells and displays separate bounds for unknown evidence. **Obstruction skyline** shows the top blocked sample and the next known open sample. **Canopy openings** groups connected decisive sky on the sphere, including north wrap when the whole bearing range is selected.
5. Enter a chronological table of measured bearing/elevation points in **Direct light path**. Paste the data or import CSV with a time zone on every timestamp. **Diffuse sky reference** and **Oriented surface sky view** report relative camera brightness support weighted by incidence direction. They do not predict daily irradiance.
6. Select two observations to **Repeat survey change** in common directions. Compare manually entered marks under **Repeat landmark agreement** and photometric patches under **Repeat brightness references**. Use **Direction transect** and **Field marks report** to revisit individual inspected directions. Field marks can be exported as JSON or CSV, then checked against the same camera bytes before importing JSON notes.
7. **Retain observation** keeps the exact sources, lens versions, settings, result images and tables. Under **Records**, reopen an inspection and read a point using its retained settings. Append a review, print the reading sheet, export a ZIP, or build a multi-inspection field folio for the station. **Read evidence** verifies the ZIP and shows its checked photographs and thumbnails without adding them to your station.

## Inputs and limits

- PNG, JPEG, WebP and single-frame TIFF, at least 24 pixels per side, up to 24 MiB and 12 megapixels. A run that selects several captures is limited to 19 million decoded pixels. EXIF orientation is applied for measurement. Supported embedded color profiles are converted to sRGB. Grayscale unsigned 16-bit data retain normalized code values. Alpha below 250 is withheld; importing an opaque picture cannot recover an occlusion that only the operator knows about.
- The camera is assumed to have a level optical axis. The four radial laws use a fixed maximum zenith and do not model lens tilt, tangential distortion or a moving station. Check residuals and field coverage before making directional claims.
- Without user samples, the sky tool uses a small fixed color reference set. The target is an interpretation of visible evidence. Cloud, glare, pale bark and exposure can require independent samples. The classifier reports an uncertainty band and withholds clipped or inadequately supported pixels.
- Angular area is a finite numerical sample. Thin openings can disappear at a coarse sampling density. Bounds account for sampled missing or ambiguous cells, not every possible error between samples. The path uses midpoint samples of arcs between entered timed positions. Raw display-code brightness is uncalibrated and does not support absolute radiance, crop recommendations or weather forecasts.
- Local limits include 28 observations and 120 retained results per station, and 220 field marks per observation. Portable evidence verification checks internal consistency and source hashes. It does not determine who authored a received file.

`crownwindow/` contains the camera, lens, physical analysis, archival and HTTP code. The browser uses `static/` only. The scoped line report can be replicated with `python3 scripts/count_backend.py --project-root . --source-root crownwindow --output /tmp/crownwindow-source-lines.json`.
