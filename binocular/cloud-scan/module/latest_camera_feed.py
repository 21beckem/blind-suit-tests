"""Live capture thread that retains only the newest stereo pair."""

from threading import Condition, Thread
import time

import cv2



class LatestCameraFeed:
    """Read the newest pair and its capture timestamp, dropping queued old pairs."""

    def __init__(self, CameraFeed, config):
        self.config = config
        self._feed = None
        self._condition = Condition()
        self._thread = None
        self._latest = None
        self._serial = 0
        self._returned = 0
        self._error = None
        self._stopping = False
        self._feed = CameraFeed(self.config).__enter__()

    def __enter__(self):
        try:
            self._feed._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._thread = Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
        except BaseException:
            self.close()
            raise
        return self

    def _capture_loop(self):
        while not self._stopping:
            try:
                left, right = self._feed.read()
                timestamp_s = time.monotonic()
            except Exception as error:
                with self._condition:
                    self._error = error
                    self._condition.notify_all()
                return
            with self._condition:
                self._latest = (left, right, timestamp_s)
                self._serial += 1
                self._condition.notify_all()

    def read(self):
        with self._condition:
            while self._serial == self._returned and self._error is None and not self._stopping:
                self._condition.wait(timeout=0.5)
            if self._serial != self._returned:
                self._returned = self._serial
                return self._latest
            if self._error is not None:
                raise self._error
            raise RuntimeError("Live camera feed is closed")

    def close(self):
        self._stopping = True
        with self._condition:
            self._condition.notify_all()
        if self._feed is not None:
            self._feed.close()
            self._feed = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __exit__(self, *_):
        self.close()
