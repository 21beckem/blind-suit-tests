"""Configuration message shared by the WebSocket stream and optional page server."""

from config import V_ON, V_OFF, ACTION_ROW_STOP, ACTION_ROW_UNKNOWN


def hud_config():
    return {
        "type": "hud-config",
        "on": V_ON,
        "off": V_OFF,
        "actionStop": ACTION_ROW_STOP,
        "actionUnknown": ACTION_ROW_UNKNOWN,
    }
