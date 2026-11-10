"""Live capture thread that retains only the newest stereo pair."""

from threading import Condition, Thread
import time

import cv2



class LatestCameraFeed:
    """Read the newest pair and its capture timestamp, dropping queued old pairs."""

    def __init__(self, CameraFeed, config):
        self._feed_class = CameraFeed
        self.config = config
        self._feed = None
        self._condition = Condition()
        self._thread = None
        self._latest = None
        self._serial = 0
        self._returned = 0
        self._error = None
        self._stopping = False
        self._last_source_frame = -1

    def __enter__(self):
        self._feed = self._feed_class(self.config).__enter__()
        try:
            self._feed._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._playback_start_s = time.monotonic()
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
                if hasattr(self._feed, "_start_time"):
                    # Real-time recorded_camera_feed.read() can return the same
                    # frame many times per source interval. Persistence must
                    # advance once per distinct source frame, not once per poll.
                    source_frame = int(self._feed._capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    if source_frame <= self._last_source_frame:
                        time.sleep(min(0.005, 0.25 / self._feed._fps))
                        continue
                    self._last_source_frame = source_frame
                    timestamp_s = source_frame / self._feed._fps
                elif hasattr(self._feed, "_frame_index"):
                    # Sequential recordings decode once in order. Pace them
                    # to source time while retaining the one-slot buffer.
                    source_frame = self._feed._frame_index - 1
                    timestamp_s = source_frame / self._feed._fps
                    delay_s = self._playback_start_s + timestamp_s - time.monotonic()
                    if delay_s > 0:
                        time.sleep(delay_s)
                else:
                    timestamp_s = time.monotonic()
            except Exception as error:
                with self._condition:
                    self._error = error
                    self._condition.notify_all()
                return
            with self._condition:
                if self._stopping:
                    return
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
