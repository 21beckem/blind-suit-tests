#!/usr/bin/env python3
"""Live CPU stereo obstacle map with a latest-frame capture thread."""

from module.benchmark import Benchmark
from module.config import load_config
from module.depth_mapper import DepthMapper
from module.ground_plane import GroundPlaneEstimator
from module.latest_camera_feed import LatestCameraFeed
from module.recorded_camera_feed_sequential import CameraFeed
from module.camera_feed import CameraFeed as PhysicalCameraFeed
from module.map_view import MapWindow
from module.obstacle_map import ObstacleMap
from module.pose_tracker import PoseTracker


def main(camera_feed_class=CameraFeed):
    config = load_config()
    depth = DepthMapper(config)
    ground = GroundPlaneEstimator(config)
    tracker = PoseTracker(config)
    grid = ObstacleMap(config)
    bench = Benchmark().setup("capture", "depth", "mapping", "display")
    try:
        with LatestCameraFeed(camera_feed_class, config["camera"]) as feed, MapWindow() as window:
            while window.is_open:
                try:
                    with bench.measure("capture"):
                        left, right, timestamp_s = feed.read()
                except RuntimeError as error:
                    if str(error) == "End of stereo recording":
                        break
                    raise
                with bench.measure("depth"):
                    result = depth.compute(left, right)
                with bench.measure("mapping"):
                    floor = ground.fit(result, timestamp_s)
                    pose = tracker.update(result, floor, timestamp_s)
                    if tracker.reset_map:
                        grid = ObstacleMap(config)
                    grid.integrate(result, floor, pose, timestamp_s)
                with bench.measure("display"):
                    window.update(result, grid)
    finally:
        bench.display()
        if config["benchmark"]["output"] is not None:
            bench.save(config["benchmark"]["output"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="store_true",
                        help="use the physical stereo camera instead of paced recording replay")
    args = parser.parse_args()
    main(PhysicalCameraFeed if args.camera else CameraFeed)
