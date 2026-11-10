#!/usr/bin/env python3
"""Load a left/right image pair, compute depth, and display it."""

from module.config import load_config
from module.depth_mapper import DepthMapper
from module.image_tools import ImageTools
from module.ui import DepthWindow


def main():
    config = load_config()
    depth = DepthMapper(config)
    left, right = ImageTools.load_pair(**config["images"])
    with DepthWindow(config["window"]["name"]) as window:
        window.update(depth.compute(left, right))
        window.wait_until_closed()


if __name__ == "__main__":
    main()
