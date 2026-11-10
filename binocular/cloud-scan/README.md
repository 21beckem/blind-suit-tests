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

Only `live_depth.py` and `image_depth.py` remain as scripts at the top level. Shared
Python code lives in the `module/` package; import it with `from module.<name> import ...`.

`CameraFeed` and `DepthWindow` are context managers: use `with` to release the camera and
close the window even when processing fails. The depth mapper is constructed once and
reused for every pair. Matching and confidence filtering preserve the original algorithm;
the uncommon tuning switches, diagnostics, and alternate cloud renders have been removed.

To capture a stereo recording, run `python -m module.recorded_camera_feed`
from the `binocular/cloud-scan` directory. The default file stays at
`cloud-scan/recorded_camera_feed.avi`. The live preview shows the whole stereo frame;
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
