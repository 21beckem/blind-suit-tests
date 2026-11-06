# Browser binocular depth test

This is the browser port of `binocular/3d-scan-tests/03_depth_test.py`. Camera
capture, the UI, depth conversion, and display run in the browser. StereoSGBM,
rectification, CLAHE, color lookup, and the epipolar check use a custom OpenCV
WebAssembly build in `wasm_construction/dist/`.

## Run

From the repository root:

```powershell
python -m http.server 8000
```

Open `http://localhost:8000/vr/binocular/` in a current desktop browser.
Camera access needs localhost or HTTPS. Select the side-by-side camera and
press **Start camera**. The page requires an actual 1280×480 stream, matching
the bundled 640×480-per-eye calibration. It checks the delivered dimensions
before processing. **Test saved pair** runs recorded frame 100 without a camera.

Space freezes the image, C toggles CLAHE, S saves settings, and Q stops the
camera. Clicking any panel measures median valid depth in a 5×5 patch. Save
settings downloads `config.json` and keeps those settings in local storage.
The page cannot silently overwrite the repository's Python config file.

The local copy of `config.json` uses 1280×480 because the existing recordings
show the camera delivered that resolution despite the original Python config's
1280×720 request. If the camera exposes only a different mode in the browser,
calibrate that mode before using its distances.

## Rebuild WASM

The checked-in `wasm_construction/dist/` assets are enough to run the page.
To reproduce them on Windows, install Visual Studio CMake/Ninja and then run
these commands from `wasm_construction/`:

```powershell
git clone --depth 1 --branch 4.12.0 https://github.com/opencv/opencv.git vendor/opencv
git clone --depth 1 https://github.com/emscripten-core/emsdk.git vendor/emsdk
Push-Location vendor/emsdk
.\emsdk.bat install 4.0.20
.\emsdk.bat activate 4.0.20
Pop-Location
.\build.ps1
node smoke.cjs
node real_pair.cjs
```

The build script enables WebAssembly SIMD and exports only the OpenCV APIs
needed here. `vendor/` and `build/` are intentionally ignored; the generated
`dist/` assets are served with the page. It adjusts two OpenCV 4.12 build
settings for Emscripten 4.0.20 (C++17 and removal of a retired linker flag).
To refresh the bundled calibration, run this from `vr/binocular/`:

```powershell
python wasm_construction/export_calibration.py ../../binocular/3d-scan-tests/calibration.npz calibration.json
```

The fixture in `wasm_construction/sample_sbs.png` comes from the repository's
recorded pair. `sample_reference.json` stores its Python depth result for a
first comparison. `real_pair.cjs` verifies WASM depth against that reference
using frame 100 in the repository recording. `prepare_fixture.py` can
regenerate the fixture and reference when NumPy and
`opencv-python-headless` are installed in `wasm_construction/vendor/pydeps`.

## Performance and scope

The page displays stage timings. The worker processes the latest available
video frame and skips new frames while busy, keeping the controls responsive.
Processing scale can be lowered when full-resolution SGBM is too slow. A
single-threaded SIMD build avoids the cross-origin isolation requirement of
WASM threads. Actual live frame rate depends on the browser, camera mode, and
CPU. This is the depth tuning test only; it does not perform 3D tracking or
point-cloud fusion.
