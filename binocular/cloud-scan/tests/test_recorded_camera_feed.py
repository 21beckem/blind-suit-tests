"""Recording and playback checks without a physical camera."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from module.config import load_config
from module.recorded_camera_feed import CameraFeed, record_video, recording_by_name
from module.recorded_camera_feed_sequential import CameraFeed as SequentialCameraFeed


class RecordedCameraFeedTests(unittest.TestCase):
    def test_record_and_replay_stereo_frames(self):
        frame = np.zeros((48, 128, 3), dtype=np.uint8)
        frame[:, 64:] = 255
        config = {"index": 1, "width": 128, "height": 48, "rotate_180": False}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture.avi"
            with patch("module.recorded_camera_feed.cv2.VideoCapture") as camera, \
                    patch("module.ui.cv2.namedWindow"), \
                    patch("module.ui.cv2.imshow") as imshow, \
                    patch("module.ui.cv2.waitKey", side_effect=[-1, -1, 27]), \
                    patch("module.ui.cv2.getWindowProperty", return_value=1), \
                    patch("module.ui.cv2.destroyWindow") as destroy:
                camera.return_value.isOpened.return_value = True
                camera.return_value.get.return_value = 30
                camera.return_value.read.return_value = (True, frame)
                self.assertEqual(record_video(config, output), output)
                camera.assert_called_once_with(1)
                self.assertEqual(camera.return_value.read.call_count, 3)
                camera.return_value.release.assert_called_once()
                self.assertEqual(imshow.call_count, 3)
                np.testing.assert_array_equal(imshow.call_args.args[1], frame)
                destroy.assert_called_once()

            self.assertGreater(output.stat().st_size, 0)
            saved = cv2.VideoCapture(str(output))
            try:
                self.assertEqual(int(saved.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            finally:
                saved.release()
            config["recording"] = str(output)
            with patch("module.recorded_camera_feed.time.monotonic", return_value=0):
                with CameraFeed(config) as feed:
                    left, right = feed.read()
            self.assertEqual(left.shape, (48, 64, 3))
            self.assertEqual(right.shape, (48, 64, 3))
            self.assertLess(float(left.mean()), 10)
            self.assertGreater(float(right.mean()), 245)

            config["rotate_180"] = True
            with patch("module.recorded_camera_feed.time.monotonic", return_value=0):
                with CameraFeed(config) as feed:
                    left, right = feed.read()
            self.assertGreater(float(left.mean()), 245)
            self.assertLess(float(right.mean()), 10)

    def test_slow_reads_skip_to_current_recording_time(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "timed.avi"
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"),
                                     30, (128, 48))
            self.assertTrue(writer.isOpened())
            try:
                for index in range(75):
                    writer.write(np.full((48, 128, 3), index, dtype=np.uint8))
            finally:
                writer.release()

            config = {"recording": str(output), "rotate_180": False}
            with patch("module.recorded_camera_feed.time.monotonic",
                       side_effect=[0, 0, 2, 2.01, 3]):
                with CameraFeed(config) as feed:
                    first, _ = feed.read()
                    after_two_seconds, _ = feed.read()
                    repeated, _ = feed.read()
                    with self.assertRaises(RuntimeError):
                        feed.read()
            self.assertLess(float(first.mean()), 2)
            self.assertAlmostEqual(float(after_two_seconds.mean()), 60, delta=2)
            np.testing.assert_array_equal(repeated, after_two_seconds)

    def test_missing_recording_fails_and_releases_capture(self):
        with patch("module.recorded_camera_feed.cv2.VideoCapture") as capture:
            capture.return_value.isOpened.return_value = False
            with self.assertRaisesRegex(RuntimeError, "Cannot open recording"):
                with CameraFeed({"recording": "missing.avi"}):
                    self.fail("An unopened recording should not enter the context")
            capture.return_value.release.assert_called_once()

    def test_default_recording_stays_in_cloud_scan(self):
        default = recording_by_name(load_config()["camera"]["recording"])
        self.assertEqual(default.parent, Path(__file__).resolve().parents[1] / "recordings")
        self.assertTrue(default.is_file())
        with patch("module.recorded_camera_feed.time.monotonic", return_value=0):
            with CameraFeed({"rotate_180": False}) as feed:
                self.assertEqual(feed.read()[0].shape, (480, 640, 3))
        with SequentialCameraFeed({"rotate_180": False}) as feed:
            self.assertEqual(feed.read()[0].shape, (480, 640, 3))

    def test_sequential_feed_never_skips_or_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sequential.avi"
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"),
                                     25, (128, 48))
            self.assertTrue(writer.isOpened())
            try:
                for index in range(5):
                    writer.write(np.full((48, 128, 3), index * 40, dtype=np.uint8))
            finally:
                writer.release()

            config = {"recording": str(output), "rotate_180": False}
            with SequentialCameraFeed(config) as feed:
                means = [float(feed.read()[0].mean()) for _ in range(5)]
                with self.assertRaisesRegex(RuntimeError, "End of stereo recording"):
                    feed.read()
            np.testing.assert_allclose(means, [0, 40, 80, 120, 160], atol=3)


if __name__ == "__main__":
    unittest.main()
