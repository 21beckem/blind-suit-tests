import numpy as np
from config import OUTPUT_WIDTH, OUTPUT_HEIGHT, V_ON, V_OFF, ACTION_ROW_STOP, ACTION_ROW_UNKNOWN


def generate_grid(decision):
    """Rows: action, imminent hazard, hazard ahead, unknown coverage."""
    if (OUTPUT_WIDTH, OUTPUT_HEIGHT) != (5, 4):
        raise ValueError("This HUD code requires a 5x4 output")
    if not 0 <= V_OFF < V_ON <= 1:
        raise ValueError("brightness must satisfy 0 <= V_OFF < V_ON <= 1")
    grid = np.full((4, 5), V_OFF, dtype=np.float32)
    if decision.action == "direction":
        grid[0, decision.direction] = V_ON
    elif decision.action == "stop":
        grid[0] = ACTION_ROW_STOP
    elif decision.action == "unknown":
        grid[0] = ACTION_ROW_UNKNOWN
    else:
        raise ValueError("unknown navigation action")
    grid[1] = np.where(decision.near, V_ON, V_OFF)
    grid[2] = np.where(decision.ahead, V_ON, V_OFF)
    grid[3] = np.where(decision.unknown, V_ON, V_OFF)
    return grid
