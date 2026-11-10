"""Click once at a fixed time, then lock the Windows workstation."""

import ctypes
import time
from datetime import datetime, timedelta


TARGET_TIME = "22:55"


def wait_until(target_time: str) -> None:
    """Wait until the next occurrence of target_time in local time."""
    scheduled = datetime.strptime(target_time, "%H:%M").time()
    now = datetime.now()
    target = datetime.combine(now.date(), scheduled)
    if target <= now:
        target += timedelta(days=1)

    while True:
        remaining = (target - datetime.now()).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 60))


def click_left_mouse_button() -> None:
    """Click at the current cursor position using the Windows API."""
    user32 = ctypes.windll.user32
    mouse_event = user32.mouse_event
    mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
    mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP


def lock_workstation() -> None:
    """Ask Windows to lock the workstation; ignore unsupported lock failures."""
    try:
        ctypes.windll.user32.LockWorkStation()
    except (AttributeError, OSError):
        pass


def main() -> None:
    print(f"Waiting until {TARGET_TIME} to click the left mouse button, then lock the workstation...")
    wait_until(TARGET_TIME)
    click_left_mouse_button()
    time.sleep(10)
    lock_workstation()


if __name__ == "__main__":
    main()