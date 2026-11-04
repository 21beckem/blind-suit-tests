# Stereo room scanner (ELP OV9281 dual-lens, Windows, CPU only)

## Setup (once)
1. Install Python 3.10-3.12 (64-bit). Double-click `setup.bat` (creates `.venv`, installs packages, runs the self-test).
2. Each session: `.venv\Scripts\activate.bat`

## Run order
| Step | Command | What it does |
|---|---|---|
| 0 | `python selftest.py` | No camera needed. Verifies install + calibration/depth/tracking/fusion math on synthetic data |
| 1 | `python 01_probe.py` | Finds the camera, sweeps modes, measures real FPS, checks left/right eye order, saves `config.json` |
| 2 | `python make_board.py`, print at 100%, glue flat. Then `python 02_calibrate.py --square-mm <measured>` | Auto-captures ~30 poses, solves intrinsics + stereo extrinsics, saves `calibration.npz` |
| 3 | `python 03_depth_test.py` | Live rectified view, epipolar check, depth map, sliders. Click a point to read distance; verify with a tape measure. `S` saves tuned settings |
| 3a | `python 05_single_frame_test.py` | Freezes one frame and saves a reproducible rectification/depth/confidence/point-cloud report before tracking is involved |
| 4 | `python 04_live_scan.py` | Live tracking + 3D map in a viewer. `S` save, `R` reset, `SPACE` pause, `Q` quit (auto-saves to `scans/`) |

Re-run step 2 whenever you change mode/resolution, swap eyes, or bump/rebuild the module.

## Single-frame validation

Run this after calibration and matcher tuning, before testing tracking:

```powershell
python 05_single_frame_test.py
```

It captures one synchronized pair and writes a timestamped directory under `validation_runs/` containing
the raw and rectified images, disparity and metric-depth arrays, a left/right consistency mask, two point
clouds, `summary.png`, and a machine-readable `report.json`. Click a textured target in the viewer to sample
a 9x9 depth patch. If its distance was measured with a tape, include it on the command line:

```powershell
python 05_single_frame_test.py --known-distance 1.19
```

For repeatable, non-interactive tests, specify a target rectangle and disable the viewer:

```powershell
python 05_single_frame_test.py --known-distance 1.19 --roi 220 150 100 80 --no-view
```

Saved pairs can be retested without the camera:

```powershell
python 05_single_frame_test.py --left left.png --right right.png --no-view
```

The report includes signed epipolar error. A positive suggested shift means the rectified right image would
need to move downward by that many pixels. `--vertical-shift N` can test that hypothesis, but it is diagnostic;
recalibration is the correct fix for a persistent offset.

## Scanning tips
- Move slowly and smoothly; rotate gently. Overlap areas you already scanned.
- Keep textured objects in view. Blank white walls/floors give no depth (holes) and weak tracking.
- Light the room well and evenly. Global shutter helps with motion blur, but it is mono and needs light.
- Closest measurable distance is printed at start (depends on baseline x focal / num_disparities).
- If you see LOST: slowly return to a place you already scanned; it re-localises automatically.

## Troubleshooting
- Probe finds nothing: other app using the camera? Try `--backend msmf`, another USB port, or `--dual 0 1`.
- FPS low at 2560x800: USB 2.0 bandwidth. Probe picks the best mode it measures; 1280x400 @120 is the fallback.
- Calibration RMS > 1.0: board not flat, wrong square size, too few varied poses, or motion blur.
- Depth scale off by a constant factor: `--square-mm` was wrong; re-run `02_calibrate.py --recalibrate --square-mm <real>`.
- Many holes: raise light, lower `uniqueness`/`speckle` in step 3, or try CLAHE (`C`).

## Known limits (honest)
- Pose tracking is keyframe RGB-D odometry with local re-localisation; there is NO global loop closure, so
  long walks around a whole room will drift. Scan in a smooth, overlapping path; for big loops expect some seam.
- Stereo depth noise grows with distance squared; above ~3-4 m it gets coarse.
- If your Open3D TSDF is unavailable the scanner falls back to voxel-averaged point fusion + Poisson meshing.
- Exports are Y-up by default (set `export_y_up` false in `config.json` for OpenCV's Y-down).
