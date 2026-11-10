# Stereo depth

Two entry scripts share the same `DepthMapper.compute(left, right)`:

```powershell
python binocular/cloud-scan/live_depth.py
python binocular/cloud-scan/image_depth.py
```

Install `numpy` and `opencv-python` in your Python environment. From the repository root,
the existing environment can also run these with `.venv/Scripts/python.exe`.
Both scripts show the rectified left image beside colorized disparity. Press **Esc**, **Q**,
or close the window to exit. Live depth measures capture, depth computation, and display,
then prints their timings and saves `stereo_out/benchmark.json` when it exits.
Image depth does not save files or run benchmarks automatically.
The old combined `stereo_depth.py` command and its flags have been replaced by these scripts
and `config.json`.

## Configuration

Edit `config.json` next to the scripts. Calibration and image paths are relative to that
JSON file, so the scripts work from any working directory. Recording paths and benchmark
output paths follow the same rule. `load_config(path)` can also load a different JSON
file from your own code. The default JSON stays in `cloud-scan/`, above `module/`.

- `calibration`: path to an OpenCV stereo `.npz` calibration. Translation must be in metres.
- `camera.index`: camera device number. Only a single side-by-side camera is supported.
- `camera.width` / `height`: requested size of the **whole** camera frame. The provided
  calibration requires 640 x 480 per eye, so the default request is 1280 x 480.
- `camera.rotate_180`: rotate the **entire captured frame before splitting** when true.
  This also exchanges the positions of the original halves. After rotation, the left
  half is always treated as the left camera; there is no automatic order detection.
  This option affects camera capture only, not images loaded from disk.
- `images.left` / `right`: paths to the two images used by `image_depth.py`.
- `depth.rectify`: true for raw frames. Set false for images that are already rectified;
  the calibration must then contain `Q`. Raw rectification uses `K1/D1/K2/D2/image_size`
  and either `R/T` or `R1/R2/P1/P2` (using or deriving `Q`).
- `depth.num_disp`: full-resolution disparity search range; the working range rounds up
  to a multiple of 16. Increase it to measure closer objects.
- `depth.block`: positive odd matching window size.
- `depth.scale`: working resolution scale; 1.0 for full size, 0.5 for half size.
- `depth.max_rel_err`: maximum expected relative depth error, used to reject far pixels.
- `depth.tex_min`: minimum texture for a pixel to be classified as solid.
- `benchmark.output`: JSON destination for live timings (the latest run replaces this
  file). Set to `null` to print timings without saving them.
- `window.name`: preview window title.

## Modules

| File | Public interface |
| --- | --- |
| `module/config.py` | `load_config(path=None)` |
| `module/depth_mapper.py` | `DepthMapper(config).compute(left, right)` -> `DepthResult` |
| `module/calibration.py` | `load_calibration(path)`, `Rectifier(calibration).rectify(left, right)` |
| `module/camera_feed.py` | `CameraFeed(camera_config)`, `read()`, `split_frame(frame, rotate_180=False)` |
| `module/recorded_camera_feed.py` | Same `CameraFeed` API, reading a saved recording |
| `module/image_tools.py` | `ImageTools.load(path)`, `load_pair(left, right)`, `save(path, image)`, `depth_image(result)` |
| `module/ui.py` | `DepthWindow(name)`, `update(result)`, `wait_until_closed()` |
| `module/point_cloud.py` | `PointCloud.from_depth(result)`, `downsample(voxel_size)`, `export(path)` |
| `module/benchmark.py` | `Benchmark`, `setup(*names)`, `start(name)`, `stop(name)`, `measure(name)`, `display()`, `save(path)` |

The depth preview scripts are `live_depth.py` and `image_depth.py`; the obstacle-map
scripts are `map_live.py` and `map_offline.py`. Shared
Python code lives in the `module/` package; import it with `from module.<name> import ...`.

`CameraFeed` and `DepthWindow` are context managers: use `with` to release the camera and
close the window even when processing fails. The depth mapper is constructed once and
reused for every pair. Matching and confidence filtering preserve the original algorithm;
the uncommon tuning switches, diagnostics, and alternate cloud renders have been removed.

To capture a stereo recording, run `python -m module.recorded_camera_feed`
from the `binocular/cloud-scan` directory. The existing default clip is at
`cloud-scan/module/recorded_camera_feed.avi`. The live preview shows the whole stereo frame;
press Esc in that window to stop and save the recording. The window's existing Q and
close-button controls also stop it. Use `--output path/to/capture.avi` to change the
destination. To replay it in
another script, replace `from module.camera_feed import CameraFeed` with
`from module.recorded_camera_feed import CameraFeed`; the constructor, context manager,
`read()`, `close()`, and `split_frame()` work the same way. Playback uses the default
file unless `camera_config["recording"]` names another recording. An optional
`camera.recording` in JSON is resolved relative to that JSON file by `load_config()`.
Playback starts when the context manager opens; each `read()` seeks to the frame at the
current elapsed time. Slow processing skips older frames, while repeated reads within
one frame interval return that frame again. Reading beyond the end of the recording
raises `RuntimeError`. The full frame is saved before rotation or splitting, so
`camera.rotate_180` behaves the same during replay.

For deterministic offline processing, use `module.recorded_camera_feed_sequential.CameraFeed`.
It has the same constructor, `read()`, context manager, `close()`, and `split_frame()`
interface, but each `read()` decodes the next frame regardless of processing speed.
`map_offline.py` uses the recording's source timestamps and reports any timestamp fallback.

## Rolling obstacle map

From this directory, run:

```powershell
..\..\.venv\Scripts\python.exe map_offline.py
..\..\.venv\Scripts\python.exe map_live.py
```

The offline command discovers compatible AVI/MP4/MOV/MKV files under `cloud-scan/`,
processes each independently, and writes `stereo_out/map_runs/summary.json` plus a
`metrics.json`, periodic top-down maps, and depth images for each recording. Wrong-size,
unreadable, low-depth, or untrackable recordings receive an explicit `unsuitable` result.
The supplied apartment recording contains 1,135 frames at 25 fps over 45.4 seconds.
Use `--recording path/to/video.avi` to process one file and `--output-dir path` to
choose a diagnostics directory; `--debug-interval-s 1` saves images every second.
`map_live.py` replays that recording sequentially at its source frame rate by default.
Pass `--camera` to use the physical stereo camera. Both paths use a one-slot capture
thread, and repeated source frames are not integrated twice. Esc or Q closes the map
window.

`config.json` has a `map` section for geometry, floor, evidence, tracking, suitability,
and output thresholds. The grid is 6 m by 6 m at 5 cm per cell. The three navigation
states are **unknown**, **free**, and **obstacle**; a below-floor warning is rendered
separately but queried as unknown. Only confidence-2 stereo points inside the trusted
3 m range affect geometry. Floor evidence is required for free space; weak/missing
depth, unobserved space, the near blind zone, and tracking loss do not become free.
Several solid floor pixels can establish a free cell in one frame. Repeated solid
floor observations reconcile older conflicting obstacle evidence. A solid floor ray
can reduce an old obstacle only where the ray crosses its measured height; it never
marks unseen ground free. Cells traversed by the wearer are cleared to unknown if
they conflict with older obstacle evidence. They are not labeled free unless floor
was actually observed there. The viewer hatches recently traversed unknown cells
in teal so the walked route remains distinct from measured free floor. Below-floor
warnings fade to unknown after they stop being observed.
Small, low horizontal stereo fragments on patterned floors and isolated
below-floor fragments remain unknown. This suppresses tile aliases and false
drop-off warnings without marking those pixels free.
Obstacle evidence now fades on a 5 s time scale: a confirmed but unobserved
obstacle remains for roughly four seconds, then becomes unknown. This reduces
old obstacle paint during a turn while retaining short-term memory. Measured
floor evidence fades on a 10 s time scale and expires after 12 s, so floor
already seen before a turn remains available longer than obstacle paint.

The camera cannot see the floor immediately around its own position. A connected
green corridor through that blind strip cannot be inferred honestly from this
recording. The top-down viewer therefore leaves it unknown even when the person
has walked through it; the map's `blind_zone()` query exposes this limitation.
Offline `metrics.json` records `near_forward` counts for the first 1 m by 1 m
ahead of the camera on every frame, including measured-floor and unknown cells.
The exact apartment recording has 3 s and 38 s checkpoints in `config.json` that
reject a return to the earlier excess obstacle and drop-off paint. Its hash
keeps the checkpoint from applying to unrelated recordings.
The floor fitter independently rejects normals more than 50 degrees from
upright, even if a sequence of wrong fits gradually rotates its prior. Mapping
pauses when the floor fit is stale; a drop-off warning also needs enough
measured floor in that frame to establish the reference surface.
Once a trustworthy floor has been found, a spatially spread prior-guided fit
can recover it through missing and aliased depth on tile or concrete grids when
ordinary RANSAC fails. Large sparse, low, nearly flat grid-line aliases remain
unknown rather than becoming obstacles or invented free cells. Initial floor
acquisition keeps the stricter RANSAC threshold.

`module.ground_plane.GroundPlaneEstimator.fit(result, timestamp_s)` returns the floor
normal, camera height, gravity transform, and fresh/usable flags.
`module.pose_tracker.PoseTracker.update(result, floor, timestamp_s)` returns world
X/Z/yaw and a tracking flag. `module.obstacle_map.ObstacleMap` integrates the frame and
offers `cell_state(x_m, z_m)`, `walked_path_mask()`,
`nearest_obstacle(direction_deg, width_deg)`, `clearance_sectors()`,
`blind_zone()`, and `tracking_ok`. The walked path is a separate layer; it does
not change the free/obstacle/unknown state. Directions are relative to
camera forward; positive degrees point right. A `None` obstacle distance means there
is no **known** obstacle in that query, not that the corridor is known free.

Camera points use OpenCV X right, Y down, Z forward. Gravity and world points use X
right, Y up, Z forward, with the floor at Y=0. The map tracks a few local metres and
decays stale evidence; it does not claim a global scan or labeled obstacle accuracy.
Tracking loss stops integration, and a long loss starts a fresh local map rather than
joining poses across an unobserved interval.
Visual tracking can bridge a floor dropout for up to 0.6 s using the last
usable floor orientation; mapping still pauses until a fresh floor returns.
Initial floor acquisition does not count as a map reset. The exact apartment
fixture checks that tracking stays continuous after acquisition and that no
reset occurs.

`DepthResult` contains `disparity` (float32 pixels, 0 invalid), `confidence` (0 invalid,
1 weak, 2 solid), `gray` (rectified left grayscale), `Q` (working-resolution reprojection
matrix), and `max_disparity`. `result.depth_m` computes a metric Z image on demand with
invalid pixels set to 0. Display colors encode disparity, rather than linear metres:
near points have warmer colors, weak matches are dimmed, and invalid pixels are dark.

## Optional image and cloud exports

Use this inside your own script in this directory, after computing `result`:

```python
from module.image_tools import ImageTools
from module.point_cloud import PointCloud

ImageTools.save("stereo_out/depth.png", ImageTools.depth_image(result))

cloud = PointCloud.from_depth(result)  # solid confidence only
cloud = cloud.downsample(0.02)        # optional: average within 2 cm voxels
cloud.export("stereo_out/points.ply")
```

Pass `min_confidence=1` to `from_depth` to include weak matches. Point coordinates are in
metres in the left-camera frame: X right, Y down, Z forward. PLY includes grayscale RGB.
Empty clouds can be downsampled and exported. `ImageTools.save` writes display images;
to preserve float depth measurements, use `numpy.save("depth_m.npy", result.depth_m)`.

## Benchmarks

`live_depth.py` already measures `capture`, `depth`, and `display` for each frame.
Display timing includes preview rendering and OpenCV event handling. A final table
and JSON export are produced on exit, including when capture or processing raises an
exception. Set `benchmark.output` to `null` to disable the export.

To use benchmarks in another script, declare the things to measure once, then place
calls around the relevant work:

```python
from module.benchmark import Benchmark

bench = Benchmark().setup("capture", "depth")

# Inside the camera loop:
bench.start("capture")
left, right = feed.read()
bench.stop("capture")

with bench.measure("depth"):
    result = depth.compute(left, right)

# When finished:
bench.display()
bench.save("stereo_out/benchmark.json")
```

Different timers can overlap. Each records count, total, mean, minimum, and maximum in
milliseconds without accumulating per-frame samples. A timer with no samples reports
null statistics (0 total); an active sample is included only after `stop()`.

## Checks

```powershell
.venv/Scripts/python.exe -m unittest discover -s binocular/cloud-scan/tests -v
```

The tests use synthetic stereo frames and mocked camera/window APIs, so no camera or
graphical desktop is required.
