#!/usr/bin/env python3
"""Read the side-by-side stereo camera, compute depth, and display it."""

from module.benchmark import Benchmark
from module.camera_feed import CameraFeed
from module.config import load_config
from module.depth_mapper import DepthMapper
from module.ui import DepthWindow


def main():
    config = load_config()
    depth = DepthMapper(config)
    bench = Benchmark().setup("capture", "depth", "display")
    try:
        with CameraFeed(config["camera"]) as feed, DepthWindow(config["window"]["name"]) as window:
            while window.is_open:
                with bench.measure("capture"):
                    left, right = feed.read()
                with bench.measure("depth"):
                    result = depth.compute(left, right)
                with bench.measure("display"):
                    window.update(result)
    finally:
        bench.display()
        if config["benchmark"]["output"] is not None:
            bench.save(config["benchmark"]["output"])


if __name__ == "__main__":
    main()
