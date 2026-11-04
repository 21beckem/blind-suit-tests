import unittest

from explorer.point_cloud_generator import load_json, resolve_config
from explorer.point_cloud_generator.config import DEFAULT_CONFIG_PATH


class ConfigTests(unittest.TestCase):
    def test_nested_overrides_preserve_defaults(self):
        config = resolve_config({"room": {"width_m": 8.0}})
        self.assertEqual(config.room.width_m, 8.0)
        self.assertEqual(config.room.depth_m, 7.0)

    def test_invalid_dropout_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_config({"scan": {"dropout": 1.0}})

    def test_defaults_match_documented_medium_values(self):
        config = resolve_config()
        self.assertEqual(config.scan.points_per_m2, 3_000)
        self.assertEqual(config.scan.noise_sigma_m, 0.008)
        self.assertEqual(config.scan.dropout, 0.03)
        self.assertEqual(config.scan.outlier_fraction, 0.001)

    def test_default_json_is_the_only_default_source(self):
        raw = load_json(DEFAULT_CONFIG_PATH)
        config = resolve_config()
        self.assertEqual(config.to_dict(), raw)

    def test_single_view_setting_overrides_default_json(self):
        config = resolve_config({"scan": {"omniscient_scan": False}})
        self.assertFalse(config.scan.omniscient_scan)


if __name__ == "__main__":
    unittest.main()

