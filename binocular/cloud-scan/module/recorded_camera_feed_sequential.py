"""Sequential recording replay with the same CameraFeed interface.

Unlike recorded_camera_feed.CameraFeed, read() advances exactly one decoded frame.
It never seeks according to processing time, so offline runs cannot skip or repeat
frames when processing is slower or faster than the recording.
"""

import cv2

from .camera_feed import CameraFeed as LiveCameraFeed
from .config import load_config
from .recorded_camera_feed import recording_by_name


class CameraFeed(LiveCameraFeed):
    """Return successive left/right pairs from a saved side-by-side recording."""

    def __enter__(self):
        recording = self._config.get("recording") or load_config()["camera"].get("recording")
        path = recording_by_name(recording)
        self._capture = cv2.VideoCapture(str(path))
        try:
            if not self._capture.isOpened():
                raise RuntimeError(f"Cannot open recording {path}")
            self._fps = self._capture.get(cv2.CAP_PROP_FPS)
            if not 1 <= self._fps <= 240:
                raise RuntimeError(f"Cannot determine frame rate of recording {path}")
            self._frame_index = 0
            self._last_timestamp_ms = None
        except BaseException:
            self.close()
            raise
        return self

    def read(self):
        if self._capture is None:
            raise RuntimeError("Open CameraFeed with a 'with' statement before reading")
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError("End of stereo recording")
        self._last_timestamp_ms = self._capture.get(cv2.CAP_PROP_POS_MSEC)
        self._frame_index += 1
        return self.split_frame(frame, self._config["rotate_180"])
