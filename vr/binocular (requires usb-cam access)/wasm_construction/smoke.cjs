/* Checks the custom OpenCV WASM exports and an 8 px synthetic stereo shift. */
const assert = require("node:assert/strict");

async function main() {
  const cv = await Promise.resolve(require("./dist/opencv.js"));
  for (const name of ["Mat", "StereoSGBM", "initUndistortRectifyMap", "remap", "CLAHE", "ORB", "BFMatcher", "applyColorMap"]) {
    assert.ok(cv[name], `Missing ${name}`);
  }
  const width = 160;
  const height = 96;
  const shift = 8;
  const left = new cv.Mat(height, width, cv.CV_8UC1);
  const right = new cv.Mat(height, width, cv.CV_8UC1);
  const disparity = new cv.Mat();
  let seed = 123456789;
  for (let i = 0; i < left.data.length; i++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    left.data[i] = seed >>> 24;
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      right.data[y * width + x] = left.data[y * width + Math.min(width - 1, x + shift)];
    }
  }
  const matcher = new cv.StereoSGBM(0, 32, 5, 200, 800, 1, 31, 10, 0, 0, cv.StereoSGBM_MODE_SGBM_3WAY);
  try {
    matcher.compute(left, right, disparity);
    const values = [];
    for (let y = 12; y < height - 12; y++) {
      for (let x = 48; x < width - 12; x++) {
        const d = disparity.data16S[y * width + x] / 16;
        if (d > 0) values.push(d);
      }
    }
    values.sort((a, b) => a - b);
    const median = values[Math.floor(values.length / 2)];
    assert.ok(values.length > 1000, `Too few valid disparities: ${values.length}`);
    assert.ok(Math.abs(median - shift) < 1, `Median disparity ${median}, expected ${shift}`);
    console.log(`SGBM OK: ${values.length} valid samples, median disparity ${median} px`);
  } finally {
    matcher.delete(); left.delete(); right.delete(); disparity.delete();
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
