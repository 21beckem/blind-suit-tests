import cv2
import numpy as np
from config import OUTPUT_WIDTH, OUTPUT_HEIGHT, DEPTH_NEAR_METERS, DEPTH_FAR_METERS


def generate_grid(depth):
    """Average valid depth brightness per cell; leave cells with no readings dark."""
    output_size = (OUTPUT_WIDTH, OUTPUT_HEIGHT)
    valid = (np.isfinite(depth) & (depth > 0)).astype(np.float32)
    brightness = depth_to_grid(depth)
    sums = cv2.resize(brightness, output_size, interpolation=cv2.INTER_AREA)
    counts = cv2.resize(valid, output_size, interpolation=cv2.INTER_AREA)
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

def depth_to_grid(depth):
    """Convert metric depth to VR brightness, with invalid pixels mapped to 0."""
    grid = np.zeros(depth.shape, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if DEPTH_FAR_METERS <= DEPTH_NEAR_METERS:
        raise ValueError("DEPTH_FAR_METERS must be greater than DEPTH_NEAR_METERS")
    grid[valid] = (DEPTH_FAR_METERS - depth[valid]) / (DEPTH_FAR_METERS - DEPTH_NEAR_METERS)
    grid[valid] = np.clip(grid[valid], 0.0, 1.0)
    return grid