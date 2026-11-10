"""OpenCV preview window, closed by Esc, Q, or its close button."""

import cv2
import numpy as np

from .image_tools import ImageTools


class DepthWindow:
    def __init__(self, name):
        self.name = name
        self.is_open = False

    def __enter__(self):
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        self.is_open = True
        return self

    def update(self, result):
        """Show the rectified left image beside its confidence-aware depth image."""
        if not self.is_open:
            return
        left = cv2.cvtColor(result.gray, cv2.COLOR_GRAY2BGR)
        cv2.imshow(self.name, np.hstack((left, ImageTools.depth_image(result))))
        self._poll(1)

    def wait_until_closed(self):
        """Keep a still image visible while allowing Esc/Q/window-close to exit."""
        while self.is_open:
            self._poll(50)

    def _poll(self, delay):
        key = cv2.waitKey(delay) & 0xFF
        if key in (27, ord("q")):
            self.close()
            return
        try:
            visible = cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = 0  # Some backends raise after the user closes the window.
        if visible < 1:
            self.close()

    def close(self):
        if self.is_open:
            self.is_open = False
            # A click on X may already have destroyed the native window.
            try:
                cv2.destroyWindow(self.name)
            except cv2.error:
                pass

    def __exit__(self, *_):
        self.close()


class RecordingWindow(DepthWindow):
    """Show whole camera frames while recording, using the shared window controls."""

    def update(self, frame):
        if not self.is_open:
            return
        cv2.imshow(self.name, frame)
        self._poll(1)
