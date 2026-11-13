"""Behavior checks for the shared pipeline and optional tools; no hardware needed."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from module.benchmark import Benchmark
from module.camera_feed import CameraFeed
from module.config import load_config
from module.depth_mapper import DepthMapper, DepthResult
from module.image_tools import ImageTools
import image_depth
import live_depth
from module.point_cloud import PointCloud
from module.ui import DepthWindow


def make_calibration(path, width=192, height=96):
    focal, baseline = 160.0, 0.06
    k = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    q = np.array([[1, 0, 0, -width / 2], [0, 1, 0, -height / 2],
                  [0, 0, 0, focal], [0, 0, 1 / baseline, 0]])
    p1 = np.column_stack((k, np.zeros(3)))
    p2 = p1.copy()
    p2[0, 3] = -focal * baseline
    np.savez(path, K1=k, D1=np.zeros(5), K2=k, D2=np.zeros(5),
             R=np.eye(3), T=np.array([-baseline, 0, 0]), R1=np.eye(3), R2=np.eye(3),
             P1=p1, P2=p2, Q=q, image_size=np.array([width, height]))
    return q


class DepthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.calibration = Path(self.temp.name) / "calibration.npz"
        make_calibration(self.calibration)
        self.config = load_config()
        self.config["calibration"] = str(self.calibration)
        self.config["depth"]["num_disp"] = 32
        rng = np.random.default_rng(17)
        self.left = rng.integers(0, 256, (96, 192), dtype=np.uint8)
        # A 12px horizontal shift represents a plane at f*B/d = 0.8 metres.
        self.right = np.zeros_like(self.left)
        self.right[:, :-12] = self.left[:, 12:]

    def test_known_depth_raw_and_rectified_at_both_scales(self):
        for rectify in (True, False):
            for scale in (1.0, 0.5):
                with self.subTest(rectify=rectify, scale=scale):
                    self.config["depth"].update(rectify=rectify, scale=scale)
                    mapper = DepthMapper(self.config)
                    result = mapper.compute(self.left, self.right)
                    cloud = PointCloud.from_depth(result)
                    self.assertGreater(len(cloud.points), 100)
                    self.assertEqual(result.disparity.shape, (int(96 * scale), int(192 * scale)))
                    self.assertAlmostEqual(float(np.median(cloud.points[:, 2])), 0.8, delta=0.025)
                    rows, cols = np.nonzero(result.confidence == 2)
                    np.testing.assert_allclose(result.depth_m[rows, cols], cloud.points[:, 2], atol=1e-6)
                    self.assertTrue(np.isfinite(result.depth_m).all())
                    # Cached mapper can process the next pair, without mutating old results.
                    np.testing.assert_array_equal(mapper.compute(self.left, self.right).disparity,
                                                  result.disparity)

    def test_blank_scene_is_valid_empty_output(self):
        result = DepthMapper(self.config).compute(np.zeros_like(self.left), np.zeros_like(self.right))
        self.assertFalse(result.disparity.any())
        self.assertFalse(result.depth_m.any())
        cloud = PointCloud.from_depth(result).downsample(0.02)
        self.assertEqual(cloud.points.shape, (0, 3))
        path = cloud.export(Path(self.temp.name) / "empty.ply")
        self.assertIn(b"element vertex 0\n", path.read_bytes())

    def test_frame_size_and_dtype_errors(self):
        mapper = DepthMapper(self.config)
        with self.assertRaisesRegex(ValueError, "same size"):
            mapper.compute(self.left, self.right[:, :-1])
        with self.assertRaisesRegex(ValueError, "calibration requires"):
            mapper.compute(self.left[:48], self.right[:48])
        with self.assertRaisesRegex(ValueError, "uint8"):
            mapper.compute(self.left.astype(np.float32), self.right)

    def test_rectification_can_derive_q_or_compute_maps(self):
        with np.load(self.calibration) as data:
            calibration = {key: data[key] for key in data.files}
        del calibration["Q"]
        np.savez(self.calibration, **calibration)
        result = DepthMapper(self.config).compute(self.left, self.right)
        self.assertAlmostEqual(float(np.median(PointCloud.from_depth(result).points[:, 2])), 0.8, delta=0.025)
        for key in ("R1", "R2", "P1", "P2"):
            del calibration[key]
        np.savez(self.calibration, **calibration)
        result = DepthMapper(self.config).compute(self.left, self.right)
        self.assertAlmostEqual(float(np.median(PointCloud.from_depth(result).points[:, 2])), 0.8, delta=0.025)


class CameraTests(unittest.TestCase):
    def test_rotate_whole_frame_before_splitting(self):
        frame = np.arange(24, dtype=np.uint8).reshape(3, 8)
        left, right = CameraFeed.split_frame(frame, rotate_180=True)
        # Original right eye becomes the rotated frame's left half.
        np.testing.assert_array_equal(left, frame[::-1, 4:][:, ::-1])
        np.testing.assert_array_equal(right, frame[::-1, :4][:, ::-1])
        left, right = CameraFeed.split_frame(frame)
        np.testing.assert_array_equal(left, frame[:, :4])
        np.testing.assert_array_equal(right, frame[:, 4:])

    def test_reject_odd_or_empty_frames(self):
        with self.assertRaisesRegex(ValueError, "even width"):
            CameraFeed.split_frame(np.zeros((3, 7), np.uint8))
        with self.assertRaises(ValueError):
            CameraFeed.split_frame(None)

    @patch("module.camera_feed.cv2.VideoCapture")
    def test_capture_failure_releases_camera(self, capture):
        capture.return_value.isOpened.return_value = True
        capture.return_value.read.return_value = (False, None)
        with self.assertRaisesRegex(RuntimeError, "Could not read"):
            with CameraFeed(load_config()["camera"]) as feed:
                feed.read()
        capture.return_value.release.assert_called_once()

    @patch("module.camera_feed.cv2.VideoCapture")
    def test_open_failure_releases_camera(self, capture):
        capture.return_value.isOpened.return_value = False
        with self.assertRaisesRegex(RuntimeError, "Cannot open"):
            with CameraFeed(load_config()["camera"]):
                self.fail("An unopened camera should not enter the context")
        capture.return_value.release.assert_called_once()


class OptionalToolsTests(unittest.TestCase):
    def test_cloud_confidence_and_ply_payload(self):
        q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 10], [0, 0, 1, 0]])
        result = DepthResult(np.array([[2, 2, 0]], np.float32),
                             np.array([[2, 1, 0]], np.uint8),
                             np.array([[80, 160, 0]], np.uint8), q, 16)
        cloud = PointCloud.from_depth(result)
        np.testing.assert_array_equal(cloud.points, [[0, 0, 5]])
        self.assertEqual(len(PointCloud.from_depth(result, min_confidence=1).points), 2)
        with tempfile.TemporaryDirectory() as temp:
            path = cloud.export(Path(temp) / "nested/cloud.ply")
            header, payload = path.read_bytes().split(b"end_header\n", 1)
            self.assertIn(b"element vertex 1\n", header)
            self.assertEqual(len(payload), 15)
            np.testing.assert_array_equal(np.frombuffer(payload[:12], dtype="<f4"), [0, 0, 5])
            self.assertEqual(payload[12:], bytes([80, 80, 80]))

    def test_voxel_average(self):
        cloud = PointCloud(np.array([[0, 0, 1], [0.01, 0, 1], [1, 0, 1]], np.float32),
                           np.array([100, 200, 50], np.uint8))
        sampled = cloud.downsample(0.1)
        np.testing.assert_allclose(sampled.points, [[0.005, 0, 1], [1, 0, 1]], atol=1e-7)
        np.testing.assert_array_equal(sampled.intensity, [150, 50])

    def test_image_round_trip_and_confidence_colors(self):
        result = DepthResult(np.array([[0, 8, 8]], np.float32),
                             np.array([[0, 1, 2]], np.uint8),
                             np.zeros((1, 3), np.uint8), np.eye(4), 16)
        image = ImageTools.depth_image(result)
        np.testing.assert_array_equal(image[0, 0], [20, 20, 20])
        np.testing.assert_array_equal(image[0, 1], (image[0, 2] * 0.35).astype(np.uint8))
        with tempfile.TemporaryDirectory() as temp:
            path = ImageTools.save(Path(temp) / "nested/depth.png", image)
            np.testing.assert_array_equal(ImageTools.load(path), image)

    def test_config_paths_are_relative_to_json(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / "settings.json"
            path.write_text(json.dumps({"calibration": "calib.npz",
                                        "images": {"left": "left.png", "right": "right.png"},
                                        "camera": {"recording": "capture.avi"},
                                        "benchmark": {"output": "results/benchmark.json"}}))
            config = load_config(path)
            self.assertEqual(config["calibration"], str(directory / "calib.npz"))
            self.assertEqual(config["images"]["left"], str(directory / "left.png"))
            self.assertEqual(config["camera"]["recording"],
                             str(directory / "recordings/capture.avi"))
            self.assertEqual(config["benchmark"]["output"], str(directory / "results/benchmark.json"))


class BenchmarkTests(unittest.TestCase):
    def test_overlap_aggregation_display_and_export(self):
        bench = Benchmark().setup("capture", "depth", "unused")
        with patch("module.benchmark.time.perf_counter", side_effect=[1, 1.125, 1.25, 1.5, 2, 2.5]):
            bench.start("capture")
            bench.start("depth")
            bench.stop("capture")
            bench.stop("depth")
            with bench.measure("capture"):
                pass
        stats = bench.results()
        self.assertEqual(stats["capture"], {"count": 2, "total_ms": 750, "min_ms": 250,
                                            "max_ms": 500, "mean_ms": 375})
        self.assertEqual(stats["depth"]["mean_ms"], 375)
        self.assertIsNone(stats["unused"]["mean_ms"])
        with redirect_stdout(io.StringIO()) as output:
            bench.display()
        self.assertIn("capture", output.getvalue())
        with tempfile.TemporaryDirectory() as temp:
            path = bench.save(Path(temp) / "nested/benchmark.json")
            self.assertEqual(json.loads(path.read_text()), stats)

    def test_timer_errors_and_exception_cleanup(self):
        bench = Benchmark().setup("depth")
        with self.assertRaises(KeyError):
            bench.start("unknown")
        with self.assertRaises(RuntimeError):
            bench.stop("depth")
        with self.assertRaisesRegex(ValueError, "processing failed"):
            with bench.measure("depth"):
                raise ValueError("processing failed")
        self.assertEqual(bench.results()["depth"]["count"], 1)
        bench.start("depth")
        with self.assertRaises(RuntimeError):
            bench.start("depth")
        with self.assertRaises(RuntimeError):
            bench.setup("new")
        bench.stop("depth")


class WindowAndEntryTests(unittest.TestCase):
    @patch("module.ui.cv2.destroyWindow")
    @patch("module.ui.cv2.namedWindow")
    @patch("module.ui.cv2.waitKey", return_value=27)
    def test_escape_closes_window(self, wait, named, destroy):
        with DepthWindow("preview") as window:
            window.wait_until_closed()
            self.assertFalse(window.is_open)
        destroy.assert_called_once_with("preview")

    @patch("module.ui.cv2.destroyWindow")
    @patch("module.ui.cv2.namedWindow")
    @patch("module.ui.cv2.waitKey", return_value=-1)
    @patch("module.ui.cv2.getWindowProperty", side_effect=cv2.error("Window was closed"))
    def test_native_close_is_handled(self, visible, wait, named, destroy):
        with DepthWindow("preview") as window:
            window.wait_until_closed()
            self.assertFalse(window.is_open)

    @patch("image_depth.DepthWindow")
    def test_image_entry_uses_real_shared_compute(self, window_class):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_calibration(root / "calibration.npz")
            left = np.random.default_rng(0).integers(0, 256, (96, 192), dtype=np.uint8)
            right = np.roll(left, -12, axis=1)
            config = load_config()
            config["calibration"] = str(root / "calibration.npz")
            config["depth"]["num_disp"] = 32
            config["images"] = {"left": str(ImageTools.save(root / "left.png", left)),
                                "right": str(ImageTools.save(root / "right.png", right))}
            with patch("image_depth.load_config", return_value=config):
                image_depth.main()
        window = window_class.return_value.__enter__.return_value
        self.assertIsInstance(window.update.call_args.args[0], DepthResult)
        window.wait_until_closed.assert_called_once()

    @patch("live_depth.DepthWindow")
    @patch("live_depth.CameraFeed")
    @patch("live_depth.DepthMapper")
    def test_live_entry_orchestrates_frame_pair(self, mapper, camera, window_class):
        window = window_class.return_value.__enter__.return_value
        window.is_open = True
        window.update.side_effect = lambda result: setattr(window, "is_open", False)
        camera.return_value.__enter__.return_value.read.return_value = ("left", "right")
        with tempfile.TemporaryDirectory() as temp:
            config = load_config()
            output = Path(temp) / "benchmark.json"
            config["benchmark"]["output"] = str(output)
            with patch("live_depth.load_config", return_value=config), redirect_stdout(io.StringIO()) as console:
                live_depth.main()
            stats = json.loads(output.read_text())
            self.assertEqual(set(stats), {"capture", "depth", "display"})
            for timer in stats.values():
                self.assertEqual(timer["count"], 1)
                self.assertGreaterEqual(timer["total_ms"], 0)
            self.assertIn("capture", console.getvalue())
        mapper.return_value.compute.assert_called_once_with("left", "right")
        window.update.assert_called_once_with(mapper.return_value.compute.return_value)
        camera.return_value.__exit__.assert_called_once()

    @patch("live_depth.DepthWindow")
    @patch("live_depth.CameraFeed")
    @patch("live_depth.DepthMapper")
    def test_live_failure_exports_timings_and_releases_camera(self, mapper, camera, window_class):
        window_class.return_value.__enter__.return_value.is_open = True
        camera.return_value.__enter__.return_value.read.side_effect = RuntimeError("Camera disconnected")
        with tempfile.TemporaryDirectory() as temp:
            config = load_config()
            output = Path(temp) / "benchmark.json"
            config["benchmark"]["output"] = str(output)
            with patch("live_depth.load_config", return_value=config), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "Camera disconnected"):
                    live_depth.main()
            stats = json.loads(output.read_text())
            self.assertEqual(stats["capture"]["count"], 1)
            self.assertEqual(stats["depth"]["count"], 0)
            self.assertEqual(stats["display"]["count"], 0)
        camera.return_value.__exit__.assert_called_once()

    @patch("live_depth.Benchmark")
    @patch("live_depth.DepthWindow")
    @patch("live_depth.CameraFeed")
    @patch("live_depth.DepthMapper")
    def test_live_benchmark_export_can_be_disabled(self, mapper, camera, window_class, benchmark):
        window_class.return_value.__enter__.return_value.is_open = False
        config = load_config()
        config["benchmark"]["output"] = None
        with patch("live_depth.load_config", return_value=config):
            live_depth.main()
        bench = benchmark.return_value.setup.return_value
        bench.display.assert_called_once()
        bench.save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
