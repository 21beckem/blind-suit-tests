import numpy as np
from config import OUTPUT_WIDTH, OUTPUT_HEIGHT


def generate_grid(decision):
    """Rows: action, imminent hazard, hazard ahead, unknown coverage."""
    if (OUTPUT_WIDTH, OUTPUT_HEIGHT) != (5, 4):
        raise ValueError("This HUD code requires a 5x4 output")
    grid = np.zeros((4, 5), dtype=np.float32)
    if decision.action == "direction":
        grid[0, decision.direction] = 1
    elif decision.action == "stop":
        grid[0] = 1
    elif decision.action == "unknown":
        grid[0, ::2] = 1
    else:
        raise ValueError("unknown navigation action")
    grid[1] = decision.near
    grid[2] = decision.ahead
    grid[3] = decision.unknown
    return grid
