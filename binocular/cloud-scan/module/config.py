"""Load JSON settings, resolving input paths relative to the JSON file."""

import json
from pathlib import Path


def load_config(path=None):
    path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / "config.json"
    path = path.resolve()
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    config["calibration"] = str((path.parent / config["calibration"]).resolve())
    for eye in ("left", "right"):
        config["images"][eye] = str((path.parent / config["images"][eye]).resolve())
    if config.get("camera", {}).get("recording") is not None:
        config["camera"]["recording"] = str((path.parent / "recordings" / config["camera"]["recording"]).resolve())
    if config.get("benchmark", {}).get("output") is not None:
        config["benchmark"]["output"] = str((path.parent / config["benchmark"]["output"]).resolve())
    if config.get("map", {}).get("output_dir") is not None:
        config["map"]["output_dir"] = str((path.parent / config["map"]["output_dir"]).resolve())
    return config
