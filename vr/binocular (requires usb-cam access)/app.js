const $ = (id) => document.getElementById(id);
const canvas = $("depth-view");
const context = canvas.getContext("2d");
const video = $("camera-video");
const status = $("status");
const cameraSelect = $("camera");
const startButton = $("start");
const sampleButton = $("sample-pair");
const freezeButton = $("freeze");
const saveButton = $("save");
const controls = $("controls");
const worker = new Worker("depth-worker.js");
const STORAGE_KEY = "binocular-depth-settings-v1";
let config;
let calibration;
let stream;
let ready = false;
let busy = false;
let frozen = false;
let frameCallback;
let processingSize;
let lastSample;
let fps = 0;
let lastFrameTime = 0;

const fields = [
  { key: "scale", label: "Processing scale", min: 0.25, max: 1, step: 0.25 },
  { key: "num_disparities", label: "Disparities", min: 32, max: 384, step: 16 },
  { key: "block_size", label: "Block size", min: 3, max: 21, step: 2 },
  { key: "uniqueness", label: "Uniqueness", min: 0, max: 30, step: 1 },
  { key: "speckle_window", label: "Speckle window", min: 0, max: 400, step: 1 },
  { key: "speckle_range", label: "Speckle range", min: 0, max: 8, step: 1 },
  { key: "min_depth", label: "Minimum depth (m)", min: 0.05, max: 3, step: 0.05 },
  { key: "max_depth", label: "Maximum depth (m)", min: 0.5, max: 10, step: 0.1 },
  { key: "edge_filter", label: "Edge filter", min: 0, max: 0.4, step: 0.01 },
];

function setStatus(message, kind = "info") {
  status.textContent = message;
  status.dataset.kind = kind;
}

function settings() {
  const values = {};
  for (const { key } of fields) values[key] = Number($(key).value);
  values.clahe = $("clahe").checked;
  values.swap_eyes = $("swap_eyes").checked;
  return values;
}

function updateField(field) {
  const value = $(field.key).value;
  $(field.key + "-value").textContent = value;
}

function renderControls() {
  controls.replaceChildren();
  for (const field of fields) {
    const label = document.createElement("label");
    label.htmlFor = field.key;
    label.textContent = field.label;
    const output = document.createElement("output");
    output.id = field.key + "-value";
    const input = document.createElement("input");
    input.type = "range";
    input.id = field.key;
    input.min = field.min;
    input.max = field.max;
    input.step = field.step;
    input.value = config.stereo[field.key];
    input.addEventListener("input", () => { updateField(field); sendSettings(); });
    label.append(output, input);
    controls.append(label);
    updateField(field);
  }
  $("clahe").checked = Boolean(config.stereo.clahe);
  $("swap_eyes").checked = Boolean(config.camera.swap_eyes);
  $("clahe").addEventListener("change", sendSettings);
  $("swap_eyes").addEventListener("change", sendSettings);
}

function sendSettings() {
  if (!ready) return;
  worker.postMessage({ type: "settings", settings: settings(), frozen });
}

async function listCameras() {
  const previous = cameraSelect.value;
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter((device) => device.kind === "videoinput");
  cameraSelect.replaceChildren(...cameras.map((device, index) => {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || `Camera ${index + 1}`;
    return option;
  }));
  if (cameras.some((device) => device.deviceId === previous)) cameraSelect.value = previous;
}

function stopCamera() {
  if (frameCallback !== undefined) video.cancelVideoFrameCallback?.(frameCallback);
  frameCallback = undefined;
  stream?.getTracks().forEach((track) => track.stop());
  stream = undefined;
  video.srcObject = null;
  busy = false;
  startButton.textContent = "Start camera";
  freezeButton.disabled = true;
  setStatus("Camera stopped.");
}

function videoFrame(_time, metadata) {
  frameCallback = video.requestVideoFrameCallback(videoFrame);
  if (!stream || !ready || busy || frozen) return;
  busy = true;
  createImageBitmap(video).then((bitmap) => {
    worker.postMessage({ type: "frame", bitmap }, [bitmap]);
  }).catch((error) => {
    busy = false;
    setStatus(`Frame capture failed: ${error.message}`, "error");
  });
  const now = performance.now();
  if (lastFrameTime) {
    const instant = 1000 / Math.max(1, now - lastFrameTime);
    fps = fps ? 0.9 * fps + 0.1 * instant : instant;
  }
  lastFrameTime = now;
  $("camera-rate").textContent = `${fps.toFixed(1)} processed fps · ${metadata.width}×${metadata.height}`;
}

async function startCamera() {
  if (stream) return stopCamera();
  if (!ready) return;
  startButton.disabled = true;
  try {
    const [w, h] = calibration.image_size;
    if (config.camera.mode !== "sbs" || config.camera.width !== 2 * w || config.camera.height !== h) {
      throw new Error(`This depth test needs a ${2 * w}×${h} side-by-side camera configuration for the bundled calibration.`);
    }
    const videoConstraints = {
      width: { exact: config.camera.width },
      height: { exact: config.camera.height },
      frameRate: { ideal: config.camera.fps },
      deviceId: cameraSelect.value ? { exact: cameraSelect.value } : undefined,
    };
    if (navigator.mediaDevices.getSupportedConstraints().resizeMode) {
      videoConstraints.resizeMode = "none";
    }
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: videoConstraints });
    const actual = stream.getVideoTracks()[0].getSettings();
    video.srcObject = stream;
    await video.play();
    if (actual.width !== 2 * w || actual.height !== h) {
      throw new Error(`Camera delivered ${actual.width}×${actual.height}; calibration needs ${2 * w}×${h}.`);
    }
    await listCameras();
    frozen = false;
    freezeButton.textContent = "Freeze";
    startButton.textContent = "Stop camera";
    freezeButton.disabled = false;
    setStatus(`Live: ${actual.width}×${actual.height} at ${actual.frameRate ?? "?"} fps reported.`);
    frameCallback = video.requestVideoFrameCallback(videoFrame);
  } catch (error) {
    stopCamera();
    setStatus(`Could not start camera: ${error.message}`, "error");
  } finally {
    startButton.disabled = false;
  }
}

async function testSavedPair() {
  if (!ready || busy) return;
  if (stream) stopCamera();
  busy = true;
  try {
    const response = await fetch("wasm_construction/sample_sbs.png");
    if (!response.ok) throw new Error("Cannot load the saved stereo pair");
    const bitmap = await createImageBitmap(await response.blob());
    frozen = true;
    worker.postMessage({ type: "frame", bitmap }, [bitmap]);
    setStatus("Showing recorded stereo pair #100. Tune settings and compare with the reference.");
  } catch (error) {
    busy = false;
    setStatus(error.message, "error");
  }
}

function saveSettings() {
  const values = settings();
  config.stereo = { ...config.stereo, ...values };
  delete config.stereo.swap_eyes;
  config.camera.swap_eyes = values.swap_eyes;
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ stereo: config.stereo, swap_eyes: config.camera.swap_eyes }));
  const file = new Blob([JSON.stringify(config, null, 2) + "\n"], { type: "application/json" });
  const url = URL.createObjectURL(file);
  const link = document.createElement("a");
  link.href = url;
  link.download = "config.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  setStatus("Settings saved in this browser and downloaded as config.json.");
}

function showSample() {
  if (!lastSample || !processingSize) return;
  const { x, y, depth } = lastSample;
  $("sample").textContent = depth === null ? `(${x}, ${y}): no depth` : `(${x}, ${y}): ${depth.toFixed(3)} m`;
  const sx = canvas.width / (processingSize.width * 2);
  const sy = canvas.height / (processingSize.height * 2);
  context.save();
  context.strokeStyle = "#fff";
  context.lineWidth = 2;
  for (const [ox, oy] of [[0, 0], [0, processingSize.height], [processingSize.width, processingSize.height]]) {
    const px = (x + ox) * sx;
    const py = (y + oy) * sy;
    context.beginPath();
    context.moveTo(px - 9, py); context.lineTo(px + 9, py);
    context.moveTo(px, py - 9); context.lineTo(px, py + 9);
    context.stroke();
  }
  context.restore();
}

worker.onmessage = ({ data }) => {
  if (data.type === "ready") {
    ready = true;
    startButton.disabled = false;
    sampleButton.disabled = false;
    saveButton.disabled = false;
    setStatus("Processing module ready. Select a camera and start.");
  } else if (data.type === "geometry") {
    processingSize = { width: data.width, height: data.height };
    canvas.width = data.width * 2;
    canvas.height = data.height * 2;
    $("geometry").textContent = `Baseline ${(data.baseline * 1000).toFixed(1)} mm · fx ${data.fx.toFixed(1)} px · processing ${data.width}×${data.height}`;
  } else if (data.type === "frame") {
    context.drawImage(data.bitmap, 0, 0);
    data.bitmap.close();
    showSample();
    busy = false;
  } else if (data.type === "stats") {
    const { epipolar, timings } = data;
    $("depth-stats").textContent = `Valid ${data.validPercent.toFixed(0)}% · closest ${(data.closestDepth).toFixed(2)} m · epipolar ${epipolar.error === null ? "n/a" : epipolar.error.toFixed(2) + " px"} (${epipolar.matches} matches)`;
    $("timings").textContent = `Rectify ${timings.rectify.toFixed(1)} ms · depth ${timings.depth.toFixed(1)} ms · total ${timings.total.toFixed(1)} ms`;
  } else if (data.type === "sample") {
    lastSample = data;
    showSample();
  } else if (data.type === "error") {
    busy = false;
    setStatus(data.message, "error");
    console.error(data.message);
  }
};

worker.onerror = (error) => {
  busy = false;
  setStatus(`Processing worker failed: ${error.message}`, "error");
};

canvas.addEventListener("click", (event) => {
  if (!processingSize) return;
  const bounds = canvas.getBoundingClientRect();
  const x = Math.floor((event.clientX - bounds.left) / bounds.width * canvas.width) % processingSize.width;
  const y = Math.floor((event.clientY - bounds.top) / bounds.height * canvas.height) % processingSize.height;
  worker.postMessage({ type: "sample", x, y });
});

startButton.addEventListener("click", startCamera);
sampleButton.addEventListener("click", testSavedPair);
freezeButton.addEventListener("click", () => {
  frozen = !frozen;
  freezeButton.textContent = frozen ? "Resume" : "Freeze";
  setStatus(frozen ? "Frame frozen. Tune the sliders on this pair." : "Live camera running.");
});
saveButton.addEventListener("click", saveSettings);
document.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
  if (event.key === " " && stream) { event.preventDefault(); freezeButton.click(); }
  if (event.key.toLowerCase() === "c") $("clahe").click();
  if (event.key.toLowerCase() === "s" && ready) saveSettings();
  if (event.key.toLowerCase() === "q" && stream) stopCamera();
});

async function boot() {
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access needs HTTPS or localhost and a supported browser.");
    const [baseConfig, baseCalibration] = await Promise.all([
      fetch("config.json").then((response) => { if (!response.ok) throw new Error("Cannot load config.json"); return response.json(); }),
      fetch("calibration.json").then((response) => { if (!response.ok) throw new Error("Cannot load calibration.json"); return response.json(); }),
    ]);
    config = baseConfig;
    calibration = baseCalibration;
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (stored?.stereo) config.stereo = { ...config.stereo, ...stored.stereo };
    if (typeof stored?.swap_eyes === "boolean") config.camera.swap_eyes = stored.swap_eyes;
    renderControls();
    await listCameras();
    worker.postMessage({ type: "init", config, calibration });
    setStatus("Loading OpenCV WebAssembly module…");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

boot();
