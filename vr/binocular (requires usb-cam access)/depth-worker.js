/* Stereo depth processing lives off the UI thread. OpenCV's SGBM runs in WASM. */
let cv;
let calibration;
let config;
let sourceSize;
let outputSize;
let maps;
let matcher;
let clahe;
let palette;
let inputCanvas;
let inputContext;
let displayCanvas;
let displayContext;
let lastFrame;
let lastDepth;
let lastEpipolar = { error: null, matches: 0 };
let frameCount = 0;

function sendError(error) {
  self.postMessage({ type: "error", message: error?.stack || String(error) });
}

async function loadOpenCV() {
  self.Module = {
    locateFile: (file) => new URL(`wasm_construction/dist/${file}`, self.location.href).href,
  };
  importScripts("wasm_construction/dist/opencv.js");
  cv = await Promise.resolve(self.cv);
  if (typeof cv.Mat !== "function") {
    await new Promise((resolve) => { cv.onRuntimeInitialized = resolve; });
  }
  for (const name of ["StereoSGBM", "initUndistortRectifyMap", "remap", "CLAHE", "ORB", "BFMatcher"]) {
    if (cv[name] === undefined) throw new Error(`Custom OpenCV WASM is missing ${name}`);
  }
}

function matrix(values) {
  const nested = Array.isArray(values[0]);
  const rows = nested ? values.length : 1;
  const cols = nested ? values[0].length : values.length;
  return cv.matFromArray(rows, cols, cv.CV_64F, values.flat(Infinity));
}

function dispose(...items) {
  for (const item of items) item?.delete?.();
}

function buildGeometry() {
  dispose(maps?.leftX, maps?.leftY, maps?.rightX, maps?.rightY);
  const [w, h] = calibration.image_size;
  sourceSize = { width: w, height: h };
  const scale = Number(config.scale);
  outputSize = {
    width: Math.max(2, Math.round(w * scale)),
    height: Math.max(2, Math.round(h * scale)),
  };
  const K1 = matrix(calibration.K1);
  const D1 = matrix(calibration.D1);
  const K2 = matrix(calibration.K2);
  const D2 = matrix(calibration.D2);
  const R1 = matrix(calibration.R1);
  const R2 = matrix(calibration.R2);
  const P1 = matrix(calibration.P1);
  const P2 = matrix(calibration.P2);
  maps = { leftX: new cv.Mat(), leftY: new cv.Mat(), rightX: new cv.Mat(), rightY: new cv.Mat() };
  try {
    const size = new cv.Size(w, h);
    cv.initUndistortRectifyMap(K1, D1, R1, P1, size, cv.CV_32FC1, maps.leftX, maps.leftY);
    cv.initUndistortRectifyMap(K2, D2, R2, P2, size, cv.CV_32FC1, maps.rightX, maps.rightY);
  } finally {
    dispose(K1, D1, K2, D2, R1, R2, P1, P2);
  }
  const sx = outputSize.width / w;
  const baseline = -calibration.P2[0][3] / calibration.P2[0][0];
  maps.fx = calibration.P1[0][0] * sx;
  maps.baseline = baseline;
  displayCanvas = new OffscreenCanvas(outputSize.width * 2, outputSize.height * 2);
  displayContext = displayCanvas.getContext("2d", { willReadFrequently: false });
  self.postMessage({ type: "geometry", ...outputSize, baseline, fx: maps.fx });
}

function buildMatcher() {
  dispose(matcher, clahe);
  const c = config;
  const nd = Math.max(16, Math.round(c.num_disparities / 16) * 16);
  const bs = Math.max(3, Math.floor(c.block_size) | 1);
  matcher = new cv.StereoSGBM(
    0, nd, bs, 8 * bs * bs, 32 * bs * bs,
    Math.floor(c.disp12_max_diff), Math.floor(c.pre_filter_cap),
    Math.floor(c.uniqueness), Math.floor(c.speckle_window),
    Math.floor(c.speckle_range), cv.StereoSGBM_MODE_SGBM_3WAY,
  );
  clahe = c.clahe ? new cv.CLAHE(2.0, new cv.Size(8, 8)) : null;
}

function makePalette() {
  const ramp = new cv.Mat(1, 256, cv.CV_8UC1);
  const colored = new cv.Mat();
  for (let i = 0; i < 256; i++) ramp.data[i] = i;
  try {
    cv.applyColorMap(ramp, colored, cv.COLORMAP_TURBO);
    const bgr = colored.data;
    palette = new Uint8Array(256 * 3);
    for (let i = 0; i < 256; i++) {
      palette[i * 3] = bgr[i * 3 + 2];
      palette[i * 3 + 1] = bgr[i * 3 + 1];
      palette[i * 3 + 2] = bgr[i * 3];
    }
  } finally {
    dispose(ramp, colored);
  }
}

function prepareInput(image) {
  if (!inputCanvas || inputCanvas.width !== image.width || inputCanvas.height !== image.height) {
    inputCanvas = new OffscreenCanvas(image.width, image.height);
    inputContext = inputCanvas.getContext("2d", { willReadFrequently: true });
  }
  inputContext.drawImage(image, 0, 0);
  return inputContext.getImageData(0, 0, image.width, image.height);
}

function rectify(image) {
  const { width: w, height: h } = sourceSize;
  if (image.width !== 2 * w || image.height !== h) {
    throw new Error(`Camera delivered ${image.width}×${image.height}; calibration requires ${2 * w}×${h}. Select that camera mode or recalibrate.`);
  }
  const rgba = cv.matFromImageData(image);
  const gray = new cv.Mat();
  const leftRect = new cv.Mat();
  const rightRect = new cv.Mat();
  let leftROI, rightROI, leftScaled, rightScaled;
  try {
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    leftROI = gray.roi(new cv.Rect(0, 0, w, h));
    rightROI = gray.roi(new cv.Rect(w, 0, w, h));
    if (config.swap_eyes) [leftROI, rightROI] = [rightROI, leftROI];
    cv.remap(leftROI, leftRect, maps.leftX, maps.leftY, cv.INTER_LINEAR, cv.BORDER_CONSTANT);
    cv.remap(rightROI, rightRect, maps.rightX, maps.rightY, cv.INTER_LINEAR, cv.BORDER_CONSTANT);
    if (outputSize.width === w && outputSize.height === h) {
      return [leftRect, rightRect];
    }
    leftScaled = new cv.Mat();
    rightScaled = new cv.Mat();
    const size = new cv.Size(outputSize.width, outputSize.height);
    cv.resize(leftRect, leftScaled, size, 0, 0, cv.INTER_AREA);
    cv.resize(rightRect, rightScaled, size, 0, 0, cv.INTER_AREA);
    dispose(leftRect, rightRect);
    return [leftScaled, rightScaled];
  } catch (error) {
    dispose(leftRect, rightRect, leftScaled, rightScaled);
    throw error;
  } finally {
    dispose(rgba, gray, leftROI, rightROI);
  }
}

function epipolarError(left, right) {
  const orb = new cv.ORB(800);
  const keysL = new cv.KeyPointVector();
  const keysR = new cv.KeyPointVector();
  const descL = new cv.Mat();
  const descR = new cv.Mat();
  const mask = new cv.Mat();
  const matches = new cv.DMatchVector();
  let bf;
  try {
    orb.detectAndCompute(left, mask, keysL, descL);
    orb.detectAndCompute(right, mask, keysR, descR);
    if (descL.empty() || descR.empty() || keysL.size() < 20 || keysR.size() < 20) {
      return { error: null, matches: 0 };
    }
    bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
    bf.match(descL, descR, matches);
    const differences = [];
    for (let i = 0; i < matches.size(); i++) {
      const match = matches.get(i);
      const pl = keysL.get(match.queryIdx).pt;
      const pr = keysR.get(match.trainIdx).pt;
      if (pl.x > pr.x && Math.abs(pl.y - pr.y) < 8) differences.push(Math.abs(pl.y - pr.y));
    }
    differences.sort((a, b) => a - b);
    const n = differences.length;
    return { error: n >= 15 ? (differences[(n - 1) >> 1] + differences[n >> 1]) / 2 : null, matches: n };
  } finally {
    dispose(orb, keysL, keysR, descL, descR, mask, matches, bf);
  }
}

function filterFlyingPixels(depth, width, height, threshold) {
  if (threshold <= 0) return;
  const bad = new Uint8Array(depth.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      const value = depth[index];
      if (!value) continue;
      let low = 1e6;
      let high = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= height) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= width) continue;
          const neighbor = depth[yy * width + xx];
          if (neighbor > 0 && neighbor < low) low = neighbor;
          if (neighbor > high) high = neighbor;
        }
      }
      if (high - low > threshold * value) bad[index] = 1;
    }
  }
  for (let i = 0; i < depth.length; i++) if (bad[i]) depth[i] = 0;
}

function colorImage(values, factor, inverted, invalid) {
  const image = new ImageData(outputSize.width, outputSize.height);
  const data = image.data;
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    const offset = i * 4;
    if (value <= invalid) {
      data[offset + 3] = 255;
      continue;
    }
    let color = Math.max(0, Math.min(255, Math.floor(value * factor)));
    if (inverted) color = 255 - color;
    data[offset] = palette[color * 3];
    data[offset + 1] = palette[color * 3 + 1];
    data[offset + 2] = palette[color * 3 + 2];
    data[offset + 3] = 255;
  }
  return image;
}

function grayImage(mat) {
  const { width: w, height: h } = outputSize;
  const image = new ImageData(w, h);
  const source = mat.data;
  const data = image.data;
  const step = Math.max(20, Math.floor(h / 12));
  for (let y = 0; y < h; y++) {
    const line = (y - Math.floor(step / 2)) % step === 0;
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const offset = i * 4;
      const value = source[i];
      data[offset] = line ? 0 : value;
      data[offset + 1] = line ? 200 : value;
      data[offset + 2] = line ? 0 : value;
      data[offset + 3] = 255;
    }
  }
  return image;
}

function render(left, right, disparities, depth) {
  const { width: w, height: h } = outputSize;
  displayContext.putImageData(grayImage(left), 0, 0);
  displayContext.putImageData(grayImage(right), w, 0);
  displayContext.putImageData(colorImage(disparities, 255 / Math.max(16, config.num_disparities), false, 0.5), 0, h);
  displayContext.putImageData(colorImage(depth, 255 / config.max_depth, true, 0), w, h);
  const bitmap = displayCanvas.transferToImageBitmap();
  self.postMessage({ type: "frame", bitmap }, [bitmap]);
}

function processFrame(image) {
  if (!cv || !maps || !matcher) return;
  const begin = performance.now();
  const [left, right] = rectify(image);
  const rectified = performance.now();
  const leftMatch = new cv.Mat();
  const rightMatch = new cv.Mat();
  const raw = new cv.Mat();
  try {
    if (clahe) {
      clahe.apply(left, leftMatch);
      clahe.apply(right, rightMatch);
    }
    matcher.compute(clahe ? leftMatch : left, clahe ? rightMatch : right, raw);
    const pixels = outputSize.width * outputSize.height;
    const disparities = new Float32Array(pixels);
    const depth = new Float32Array(pixels);
    const fixed = raw.data16S;
    const fb = maps.fx * maps.baseline;
    let valid = 0;
    for (let i = 0; i < pixels; i++) {
      const disparity = fixed[i] / 16;
      disparities[i] = disparity;
      if (disparity <= 0.5) continue;
      const z = fb / disparity;
      if (z >= config.min_depth && z <= config.max_depth) depth[i] = z;
    }
    filterFlyingPixels(depth, outputSize.width, outputSize.height, config.edge_filter);
    for (let i = 0; i < pixels; i++) if (depth[i] > 0) valid++;
    lastDepth = depth;
    const matched = performance.now();
    if (++frameCount % 20 === 1) {
      try { lastEpipolar = epipolarError(left, right); }
      catch (error) { lastEpipolar = { error: null, matches: 0 }; }
    }
    render(left, right, disparities, depth);
    self.postMessage({
      type: "stats", validPercent: valid * 100 / pixels,
      closestDepth: fb / Math.max(16, config.num_disparities),
      epipolar: lastEpipolar, timings: {
        rectify: rectified - begin,
        depth: matched - rectified,
        total: performance.now() - begin,
      },
    });
  } finally {
    dispose(left, right, leftMatch, rightMatch, raw);
  }
}

function sampleDepth(x, y) {
  if (!lastDepth) return;
  const { width: w, height: h } = outputSize;
  const values = [];
  for (let yy = Math.max(0, y - 2); yy <= Math.min(h - 1, y + 2); yy++) {
    for (let xx = Math.max(0, x - 2); xx <= Math.min(w - 1, x + 2); xx++) {
      const value = lastDepth[yy * w + xx];
      if (value > 0) values.push(value);
    }
  }
  values.sort((a, b) => a - b);
  const n = values.length;
  self.postMessage({ type: "sample", x, y, depth: n ? (values[(n - 1) >> 1] + values[n >> 1]) / 2 : null });
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") {
      calibration = data.calibration;
      config = { ...data.config.stereo, swap_eyes: data.config.camera.swap_eyes };
      await loadOpenCV();
      buildGeometry();
      buildMatcher();
      makePalette();
      self.postMessage({ type: "ready" });
    } else if (data.type === "settings") {
      const scaleChanged = Number(data.settings.scale) !== Number(config.scale);
      config = { ...config, ...data.settings };
      if (scaleChanged) buildGeometry();
      buildMatcher();
      if (data.frozen && lastFrame) processFrame(lastFrame);
    } else if (data.type === "frame") {
      lastFrame = prepareInput(data.bitmap);
      data.bitmap.close();
      processFrame(lastFrame);
    } else if (data.type === "sample") {
      sampleDepth(data.x, data.y);
    }
  } catch (error) {
    data.bitmap?.close?.();
    sendError(error);
  }
};
