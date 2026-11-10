"""One side-by-side stereo camera, with optional whole-frame rotation."""

import cv2


class CameraFeed:
    def __init__(self, config):
        self._config = config
        self._capture = None

    @staticmethod
    def split_frame(frame, rotate_180=False):
        """Rotate the ENTIRE frame first, then return its left and right halves."""
        if frame is None or frame.ndim not in (2, 3) or frame.size == 0:
            raise ValueError("Camera returned an empty or invalid frame")
        if frame.shape[1] % 2:
            raise ValueError("Side-by-side camera frame must have an even width")
        if rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        middle = frame.shape[1] // 2
        return frame[:, :middle], frame[:, middle:]

    def __enter__(self):
        self._capture = cv2.VideoCapture(self._config["index"])
        try:
            if not self._capture.isOpened():
                raise RuntimeError(f"Cannot open camera {self._config['index']}")
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config["width"])
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config["height"])
        except BaseException:
            self.close()
            raise
        return self

    def read(self):
        if self._capture is None:
            raise RuntimeError("Open CameraFeed with a 'with' statement before reading")
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError("Could not read a frame from the stereo camera")
        return self.split_frame(frame, self._config["rotate_180"])

    def close(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __exit__(self, *_):
        self.close()
