import importlib.util
from pathlib import Path
import unittest

import open3d as o3d

from explorer.point_cloud_generator import generate_point_cloud


class DetectorCompatibilityTests(unittest.TestCase):
    def test_existing_detector_accepts_generated_cloud(self):
        root = Path(__file__).resolve().parents[3]
        detector_path = root / "binocular" / "with-phone" / "obstacle_scene.py"
        spec = importlib.util.spec_from_file_location("obstacle_scene_compat_test", detector_path)
        detector = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(detector)
        points = generate_point_cloud(
            {
                "seed": 2,
                "room": {"width_m": 3.5, "depth_m": 4.5, "height_m": 2.4},
                "objects": {"count": 2},
                "scan": {
                    "points_per_m2": 500,
                    "max_points": 60_000,
                    "noise_sigma_m": 0.001,
                    "flat_plane_degradation": False,
                },
            }
        )
        config = detector.default_config(
            {"min_plane_points": 400, "min_plane_area_m2": 0.20, "min_cluster_points": 40}
        )
        result = detector.analyze_point_cloud(o3d, points, config)
        self.assertIn("floor", result)
        self.assertGreaterEqual(len(result["walls"]), 1)


if __name__ == "__main__":
    unittest.main()

