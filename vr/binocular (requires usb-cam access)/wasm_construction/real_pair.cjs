/* Compare the custom WASM module against Python on recorded frame 100. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

function matrix(cv, values) {
  const nested = Array.isArray(values[0]);
  return cv.matFromArray(nested ? values.length : 1,
    nested ? values[0].length : values.length, cv.CV_64F, values.flat(Infinity));
}

function removeFlying(depth, w, h, threshold) {
  const bad = new Uint8Array(depth.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (depth[i] <= 0) continue;
      let min = 1e6;
      let max = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          const yy = y + dy;
          if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
          const value = depth[yy * w + xx];
          if (value > 0 && value < min) min = value;
          if (value > max) max = value;
        }
      }
      if (max - min > threshold * depth[i]) bad[i] = 1;
    }
  }
  for (let i = 0; i < depth.length; i++) if (bad[i]) depth[i] = 0;
}

async function main() {
  const root = path.resolve(__dirname, "../../..");
  const source = path.join(root, "binocular/3d-scan-tests");
  const record = path.join(source, "recordings/recording_dining_table");
  const cv = await Promise.resolve(require("./dist/opencv.js"));
  const calib = JSON.parse(fs.readFileSync(path.join(__dirname, "../calibration.json")));
  const cfg = JSON.parse(fs.readFileSync(path.join(__dirname, "../config.json")));
  const reference = JSON.parse(fs.readFileSync(path.join(__dirname, "sample_reference.json")));
  const [w, h] = calib.image_size;
  const raw = Buffer.alloc(2 * w * h);
  const handle = fs.openSync(path.join(record, "frames.bin"), "r");
  try { assert.equal(fs.readSync(handle, raw, 0, raw.length, (reference.frame_id - 1) * raw.length), raw.length); }
  finally { fs.closeSync(handle); }

  const objects = [];
  const own = (value) => { objects.push(value); return value; };
  try {
    const left = own(new cv.Mat(h, w, cv.CV_8UC1));
    const right = own(new cv.Mat(h, w, cv.CV_8UC1));
    left.data.set(raw.subarray(0, w * h));
    right.data.set(raw.subarray(w * h));
    const rl = own(new cv.Mat());
    const rr = own(new cv.Mat());
    for (const [eye, src, dst] of [["1", left, rl], ["2", right, rr]]) {
      const mx = own(new cv.Mat());
      const my = own(new cv.Mat());
      cv.initUndistortRectifyMap(
        own(matrix(cv, calib[`K${eye}`])), own(matrix(cv, calib[`D${eye}`])),
        own(matrix(cv, calib[`R${eye}`])), own(matrix(cv, calib[`P${eye}`])),
        new cv.Size(w, h), cv.CV_32FC1, mx, my,
      );
      cv.remap(src, dst, mx, my, cv.INTER_LINEAR, cv.BORDER_CONSTANT);
    }
    const clahe = own(new cv.CLAHE(2.0, new cv.Size(8, 8)));
    const enhanced = own(new cv.Mat());
    clahe.apply(rl, enhanced);
    assert.equal(enhanced.rows, h);
    const orb = own(new cv.ORB(800));
    const keysL = own(new cv.KeyPointVector());
    const keysR = own(new cv.KeyPointVector());
    const descL = own(new cv.Mat());
    const descR = own(new cv.Mat());
    const mask = own(new cv.Mat());
    orb.detectAndCompute(rl, mask, keysL, descL);
    orb.detectAndCompute(rr, mask, keysR, descR);
    assert.ok(keysL.size() >= 20 && keysR.size() >= 20);
    const bf = own(new cv.BFMatcher(cv.NORM_HAMMING, true));
    const matches = own(new cv.DMatchVector());
    bf.match(descL, descR, matches);
    assert.ok(matches.size() >= 15);
    const vertical = [];
    for (let i = 0; i < matches.size(); i++) {
      const match = matches.get(i);
      const a = keysL.get(match.queryIdx).pt;
      const b = keysR.get(match.trainIdx).pt;
      if (a.x > b.x && Math.abs(a.y - b.y) < 8) vertical.push(Math.abs(a.y - b.y));
    }
    assert.ok(vertical.length >= 15);
    console.log(`CLAHE and epipolar APIs OK: ${vertical.length} filtered ORB matches`);
    const s = cfg.stereo;
    const bs = s.block_size;
    const matcher = own(new cv.StereoSGBM(0, s.num_disparities, bs, 8 * bs * bs,
      32 * bs * bs, s.disp12_max_diff, s.pre_filter_cap, s.uniqueness,
      s.speckle_window, s.speckle_range, cv.StereoSGBM_MODE_SGBM_3WAY));
    const disp = own(new cv.Mat());
    const begin = performance.now();
    matcher.compute(rl, rr, disp);
    const computeMs = performance.now() - begin;
    const baseline = -calib.P2[0][3] / calib.P2[0][0];
    const fb = calib.P1[0][0] * baseline;
    const depth = new Float32Array(w * h);
    const values = disp.data16S;
    for (let i = 0; i < depth.length; i++) {
      const d = values[i] / 16;
      if (d <= 0.5) continue;
      const z = fb / d;
      if (z >= s.min_depth && z <= s.max_depth) depth[i] = z;
    }
    removeFlying(depth, w, h, s.edge_filter);
    const valid = Array.from(depth).filter((value) => value > 0).sort((a, b) => a - b);
    const validPercent = valid.length * 100 / depth.length;
    const median = valid[Math.floor(valid.length / 2)];
    console.log(JSON.stringify({ computeMs, validPercent, median, python: reference }, null, 2));
    assert.ok(Math.abs(validPercent - reference.valid_percent) < 0.2,
      "WASM valid depth differs from the Python reference by at least 0.2 percentage points");
    assert.ok(Math.abs(median - reference.median_depth_m) < 0.02,
      "WASM median depth differs from the Python reference by at least 0.02 m");
  } finally {
    for (const object of objects.reverse()) object.delete();
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
