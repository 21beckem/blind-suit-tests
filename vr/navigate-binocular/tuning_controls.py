"""Live OpenCV sliders for the walking policy and obstacle sensitivity."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from config import (
    DEPTH_FAR_METERS, PATH_WIDTH_METERS, STOP_DISTANCE_METERS,
    NEAR_LIMIT_METERS, MIN_OBSERVED_FRACTION, MIN_HAZARD_POINTS,
    DEBOUNCE_ADD_OBSTACLE_MS, DEBOUNCE_REMOVE_OBSTACLE_MS,
    DEBOUNCE_ADD_UNKNOWN_MS, DEBOUNCE_REMOVE_UNKNOWN_MS,
    DEBOUNCE_ADD_PATH_MS, DEBOUNCE_PATH_CHOICE_MS,
    DEBOUNCE_ADD_STOP_MS, DEBOUNCE_RELEASE_STOP_MS,
)


WINDOW = "Navigation Controls"
SETTINGS_PATH = Path(__file__).resolve().parent / "navigation_settings.json"


@dataclass(frozen=True)
class TuningValues:
    stop_distance: float = STOP_DISTANCE_METERS
    near_distance: float = NEAR_LIMIT_METERS
    ahead_distance: float = DEPTH_FAR_METERS
    path_width: float = PATH_WIDTH_METERS
    min_coverage: float = MIN_OBSERVED_FRACTION
    hazard_points: int = MIN_HAZARD_POINTS
    debounce_add_obstacle_ms: int = DEBOUNCE_ADD_OBSTACLE_MS
    debounce_remove_obstacle_ms: int = DEBOUNCE_REMOVE_OBSTACLE_MS
    debounce_add_unknown_ms: int = DEBOUNCE_ADD_UNKNOWN_MS
    debounce_remove_unknown_ms: int = DEBOUNCE_REMOVE_UNKNOWN_MS
    debounce_add_path_ms: int = DEBOUNCE_ADD_PATH_MS
    debounce_path_choice_ms: int = DEBOUNCE_PATH_CHOICE_MS
    debounce_add_stop_ms: int = DEBOUNCE_ADD_STOP_MS
    debounce_release_stop_ms: int = DEBOUNCE_RELEASE_STOP_MS

    def __post_init__(self):
        if not (0.4 <= self.stop_distance <= 2.5
                and self.stop_distance + 0.2 <= self.near_distance + 1e-9 <= 3.2 + 1e-9
                and self.near_distance + 0.2 <= self.ahead_distance + 1e-9 <= 4.0 + 1e-9
                and 0.4 <= self.path_width <= 1.2
                and 0.05 <= self.min_coverage <= 0.6
                and 1 <= self.hazard_points <= 30
                and all(isinstance(value, int) and 0 <= value <= 2000
                        for value in (self.debounce_add_obstacle_ms,
                                      self.debounce_remove_obstacle_ms,
                                      self.debounce_add_unknown_ms,
                                      self.debounce_remove_unknown_ms,
                                      self.debounce_add_path_ms,
                                      self.debounce_path_choice_ms,
                                      self.debounce_add_stop_ms,
                                      self.debounce_release_stop_ms))):
            raise ValueError("navigation settings are outside supported ranges")

    def navigator_kwargs(self):
        return dict(
            path_width=self.path_width,
            stop_distance=self.stop_distance,
            near_limit=self.near_distance,
            ahead_limit=self.ahead_distance,
            min_observed_fraction=self.min_coverage,
            debounce_add_obstacle_ms=self.debounce_add_obstacle_ms,
            debounce_remove_obstacle_ms=self.debounce_remove_obstacle_ms,
            debounce_add_unknown_ms=self.debounce_add_unknown_ms,
            debounce_remove_unknown_ms=self.debounce_remove_unknown_ms,
            debounce_add_path_ms=self.debounce_add_path_ms,
            debounce_path_choice_ms=self.debounce_path_choice_ms,
            debounce_add_stop_ms=self.debounce_add_stop_ms,
            debounce_release_stop_ms=self.debounce_release_stop_ms,
        )


def settings_from_dict(data):
    """Load captures/settings made before frame-based switching was removed."""
    data = dict(data)
    data.pop("switch_frames", None)
    return TuningValues(**data)


def load_settings(path=SETTINGS_PATH):
    if not path.exists():
        return TuningValues()
    try:
        return settings_from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as exc:
        print(f"Ignoring invalid navigation settings: {exc}", flush=True)
        return TuningValues()


def save_settings(values, path=SETTINGS_PATH):
    path.write_text(json.dumps(asdict(values), indent=2) + "\n", encoding="utf-8")


class NavigationControls:
    TRACKBARS = (
        ("Stop distance cm", "stop_distance", 100, 250),
        ("Near distance cm", "near_distance", 100, 320),
        ("Ahead distance cm", "ahead_distance", 100, 400),
        ("Path width cm", "path_width", 100, 120),
        ("Min coverage percent", "min_coverage", 100, 60),
        ("Obstacle points per cell", "hazard_points", 1, 30),
        ("Add obstacle ms", "debounce_add_obstacle_ms", 1, 2000),
        ("Remove obstacle ms", "debounce_remove_obstacle_ms", 1, 2000),
        ("Add unknown ms", "debounce_add_unknown_ms", 1, 2000),
        ("Remove unknown ms", "debounce_remove_unknown_ms", 1, 2000),
        ("Add path ms", "debounce_add_path_ms", 1, 2000),
        ("Path choice ms", "debounce_path_choice_ms", 1, 2000),
        ("Add stop ms", "debounce_add_stop_ms", 1, 2000),
        ("Release stop ms", "debounce_release_stop_ms", 1, 2000),
    )

    def __init__(self, values):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        instructions = np.full((100, 700, 3), (30, 30, 30), np.uint8)
        cv2.putText(instructions, "Changes apply to the next frame", (12, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(instructions, "s: save   d: defaults   r: capture   q/Esc: quit", (12, 67),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, instructions)
        for label, field, scale, maximum in self.TRACKBARS:
            cv2.createTrackbar(label, WINDOW, round(getattr(values, field) * scale), maximum,
                               lambda _position: None)
        cv2.resizeWindow(WINDOW, 700, 760)

    def reset(self):
        values = TuningValues()
        for label, field, scale, _maximum in self.TRACKBARS:
            cv2.setTrackbarPos(label, WINDOW, round(getattr(values, field) * scale))
        return values

    def read(self):
        raw = {field: cv2.getTrackbarPos(label, WINDOW) / scale
               for label, field, scale, _maximum in self.TRACKBARS}
        stop = max(0.4, raw["stop_distance"])
        near = max(stop + 0.2, raw["near_distance"])
        ahead = max(near + 0.2, raw["ahead_distance"])
        values = TuningValues(
            stop_distance=round(stop, 2), near_distance=round(near, 2),
            ahead_distance=round(ahead, 2),
            path_width=round(max(0.4, raw["path_width"]), 2),
            min_coverage=round(max(0.05, raw["min_coverage"]), 2),
            hazard_points=max(1, int(raw["hazard_points"])),
            debounce_add_obstacle_ms=int(raw["debounce_add_obstacle_ms"]),
            debounce_remove_obstacle_ms=int(raw["debounce_remove_obstacle_ms"]),
            debounce_add_unknown_ms=int(raw["debounce_add_unknown_ms"]),
            debounce_remove_unknown_ms=int(raw["debounce_remove_unknown_ms"]),
            debounce_add_path_ms=int(raw["debounce_add_path_ms"]),
            debounce_path_choice_ms=int(raw["debounce_path_choice_ms"]),
            debounce_add_stop_ms=int(raw["debounce_add_stop_ms"]),
            debounce_release_stop_ms=int(raw["debounce_release_stop_ms"]),
        )
        # Keep the sliders honest when a dependent distance is raised.
        for label, field, scale, _maximum in self.TRACKBARS:
            position = round(getattr(values, field) * scale)
            if cv2.getTrackbarPos(label, WINDOW) != position:
                cv2.setTrackbarPos(label, WINDOW, position)
        return values
