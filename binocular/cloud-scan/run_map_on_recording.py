import curses
from pathlib import Path

import cv2

import map_live
from module.config import load_config


RECORDINGS_DIR = Path(__file__).parent / "recordings"
RAW_WINDOW_NAME = "Raw video"


def play_raw_recording(recording, rotate_180):
    capture = cv2.VideoCapture(str(recording))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open recording {recording}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not 1 <= fps <= 240:
        capture.release()
        raise RuntimeError(f"Cannot determine frame rate of recording {recording}")

    cv2.namedWindow(RAW_WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        delay_ms = max(1, round(1000 / fps))
        while True:
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                break
            if rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            cv2.imshow(RAW_WINDOW_NAME, frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord("q")):
                break
            try:
                if cv2.getWindowProperty(RAW_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        capture.release()
        try:
            cv2.destroyWindow(RAW_WINDOW_NAME)
        except cv2.error:
            pass


def select_recording(stdscr):
    recordings = sorted(RECORDINGS_DIR.glob("*.avi"))
    camera_config = load_config()["camera"]
    selected = 0
    last_selected_name = None
    status = ""

    while True:
        stdscr.erase()
        stdscr.addstr(0, 0, "Select a recording (Enter to run, r for raw video, q to quit)")
        height, width = stdscr.getmaxyx()

        if not recordings:
            stdscr.addstr(2, 0, f"No .avi files found in {RECORDINGS_DIR}")
        else:
            visible_count = max(1, height - 3)
            first_visible = min(
                selected, max(0, len(recordings) - visible_count)
            )
            for row, recording in enumerate(
                recordings[first_visible:first_visible + visible_count], start=1
            ):
                name = recording.stem
                prefix = "> " if first_visible + row - 1 == selected else "  "
                stdscr.addnstr(row, 0, prefix + name, max(1, width - 1))

        if status:
            stdscr.addnstr(height - 1, 0, status, max(1, width - 1))
        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            return
        if not recordings:
            continue
        if key in (curses.KEY_UP, ord("k")):
            selected = max(0, selected - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected = min(len(recordings) - 1, selected + 1)
        elif key in (ord("r"), ord("R")):
            recording = recordings[selected]
            try:
                play_raw_recording(recording, camera_config["rotate_180"])
            except Exception as error:
                status = f"{type(error).__name__}: {error}"
            else:
                status = "Raw playback finished."
            stdscr.touchwin()
            stdscr.clear()
        elif key in (curses.KEY_ENTER, 10, 13):
            recording = recordings[selected]
            last_selected_name = recording.name
            try:
                map_live.main(recorded_feed_name=recording.stem)
            except Exception as error:
                status = f"{type(error).__name__}: {error}"
            else:
                status = "Recording finished."
            selected = next(
                (
                    index
                    for index, item in enumerate(recordings)
                    if item.name == last_selected_name
                ),
                selected,
            )
            stdscr.touchwin()
            stdscr.clear()


if __name__ == "__main__":
    curses.wrapper(select_recording)