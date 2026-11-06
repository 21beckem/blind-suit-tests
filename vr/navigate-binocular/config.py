OUTPUT_WIDTH = 5
OUTPUT_HEIGHT = 4

# Brightness values sent to the phone for every HUD cell.
V_ON = 0.5
V_OFF = 0.0
if not 0.0 <= V_OFF < V_ON <= 1.0:
    raise ValueError("HUD brightness must satisfy 0 <= V_OFF < V_ON <= 1")

def row(rowStr):
    # Convert a string of spaces and Xs to a list of floats for the HUD.
    return [V_ON if c == "X" else V_OFF for c in rowStr]

ACTION_ROW_STOP =    row("XXXXX")
ACTION_ROW_UNKNOWN = row("     ")
if any(len(row) != 5 or any(value not in (V_ON, V_OFF) for value in row)
       for row in (ACTION_ROW_STOP, ACTION_ROW_UNKNOWN)):
    raise ValueError("action rows must each contain five HUD brightness values")

DEPTH_NEAR_METERS = 0.20
DEPTH_FAR_METERS = 3.0
MAX_AHEAD_METERS = 4.0

# Initial walking prototype; tune from recorded and supervised trials.
STOP_DISTANCE_METERS = 0.5
PATH_WIDTH_METERS = 0.6
NEAR_LIMIT_METERS = 2.1
MIN_OBSERVED_FRACTION = 0.05
MIN_HAZARD_POINTS = 5

# Hold a changed observation continuously for this many milliseconds before
# showing it. While a possible STOP is being checked, the action is UNKNOWN.
DEBOUNCE_ADD_OBSTACLE_MS = 250
DEBOUNCE_REMOVE_OBSTACLE_MS = 400
DEBOUNCE_ADD_UNKNOWN_MS = 100
DEBOUNCE_REMOVE_UNKNOWN_MS = 300
DEBOUNCE_ADD_PATH_MS = 250
DEBOUNCE_PATH_CHOICE_MS = 350
DEBOUNCE_ADD_STOP_MS = 50
DEBOUNCE_RELEASE_STOP_MS = 400
