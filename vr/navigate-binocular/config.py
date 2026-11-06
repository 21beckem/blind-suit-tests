OUTPUT_WIDTH = 5
OUTPUT_HEIGHT = 4

# Brightness values sent to the phone for every HUD cell.
V_ON = 0.5
V_OFF = 0.0
if not 0.0 <= V_OFF < V_ON <= 1.0:
    raise ValueError("HUD brightness must satisfy 0 <= V_OFF < V_ON <= 1")

DEPTH_NEAR_METERS = 0.20
DEPTH_FAR_METERS = 3.0
MAX_AHEAD_METERS = 4.0

# Initial walking prototype; tune from recorded and supervised trials.
STOP_DISTANCE_METERS = 0.5
PATH_WIDTH_METERS = 0.6
NEAR_LIMIT_METERS = 2.1
MIN_OBSERVED_FRACTION = 0.05
MIN_HAZARD_POINTS = 5
SWITCH_FRAMES = 3
