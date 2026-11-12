#!/usr/bin/env python3
"""Record the stereo camera, or replay its recording with CameraFeed's API."""

import argparse
from pathlib import Path
import sys
import time

import cv2

if __package__:
    from .camera_feed import CameraFeed as LiveCameraFeed
    from .config import load_config
    from .ui import RecordingWindow
else:
    # Also support running this file by path, outside the package import context.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from module.camera_feed import CameraFeed as LiveCameraFeed
    from module.config import load_config
    from module.ui import RecordingWindow


def recording_by_name(name: str) -> Path:
    """Return the path to a recording file in the module/recordings directory."""
    if isinstance(name, Path):
        return name
    if not name:
        raise ValueError("Recording name must be non-empty")
    if not name.endswith(".avi"):
        name += ".avi"
    return (Path(__file__).parent.parent / "recordings" / name).resolve()



class CameraFeed(LiveCameraFeed):
    """Read a saved recording in real time as left/right stereo pairs.

    The camera config has the same required keys as the live feed. An optional
    ``recording`` path selects a file other than the default in cloud-scan/.
    """

    def __enter__(self):
        path = recording_by_name(self._config.get("recording"))
        self._capture = cv2.VideoCapture(str(path))
        try:
            if not self._capture.isOpened():
                raise RuntimeError(f"Cannot open recording {path}")
            self._fps = self._capture.get(cv2.CAP_PROP_FPS)
            if not 1 <= self._fps <= 240:
                raise RuntimeError(f"Cannot determine frame rate of recording {path}")
            self._start_time = time.monotonic()
        except BaseException:
            self.close()
            raise
        return self

    def read(self):
        if self._capture is None:
            raise RuntimeError("Open CameraFeed with a 'with' statement before reading")
        frame_index = int((time.monotonic() - self._start_time) * self._fps)
        if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise RuntimeError("Could not seek within the stereo recording")
        return super().read()


def record_video(config, output: Path) -> Path:
    """Save whole stereo frames until the preview window is closed."""
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(config["index"])
    writer = None
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open camera {config['index']}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not 1 <= fps <= 240:
            fps = 30.0

        with RecordingWindow("Stereo recording (Esc to stop)") as preview:
            while preview.is_open:
                ok, frame = capture.read()
                if not ok or frame is None or frame.size == 0:
                    raise RuntimeError("Could not read a frame from the stereo camera")
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output), cv2.VideoWriter.fourcc(*"MJPG"), fps,
                        (width, height), frame.ndim == 3,
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create recording {output}")
                writer.write(frame)
                display = cv2.rotate(frame, cv2.ROTATE_180) if config["rotate_180"] else frame
                preview.update(display)
    finally:
        if writer is not None:
            writer.release()
        capture.release()
    return output


def main():
    config = load_config()
    default_recording = recording_by_name(config["camera"].get("recording"))
    parser = argparse.ArgumentParser(description="Record the side-by-side stereo camera")
    parser.add_argument("-o", "--output", type=str, default=default_recording,
                        help="AVI file to write (default: in the cloud-scan directory)")
    args = parser.parse_args()

    output_file = recording_by_name(args.output)
    if output_file.exists():
        answer = input(f"Recording {output_file.name} already exists. Overwrite? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("Recording cancelled.")
            return

    output = record_video(config["camera"], output_file)
    print(f"Saved recording to {output}")


if __name__ == "__main__":
    main()
