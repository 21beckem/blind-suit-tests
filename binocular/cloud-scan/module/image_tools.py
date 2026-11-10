"""Image loading, saving, and depth visualization; no windows or capture."""

from pathlib import Path

import cv2
import numpy as np


class ImageTools:
    @staticmethod
    def load(path):
        """Load an image as uint8 BGR, suitable for DepthMapper.compute()."""
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return image

    @staticmethod
    def load_pair(left, right):
        return ImageTools.load(left), ImageTools.load(right)

    @staticmethod
    def save(path, image):
        """Save an image, creating its parent directory when needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Cannot save image: {path}")
        return path

    @staticmethod
    def depth_image(result):
        """Color disparity: near is warm, far is cool; weak pixels dim, invalid dark."""
        values = np.clip(result.disparity * (255 / result.max_disparity), 0, 255).astype(np.uint8)
        image = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
        weak = result.confidence == 1
        image[weak] = (image[weak] * 0.35).astype(np.uint8)
        image[result.disparity == 0] = (20, 20, 20)
        return image
