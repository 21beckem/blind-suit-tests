import curses
from pathlib import Path

import map_live


RECORDINGS_DIR = Path(__file__).parent / "recordings"


def select_recording(stdscr):
    recordings = sorted(RECORDINGS_DIR.glob("*.avi"))
    selected = 0
    last_selected_name = None
    status = ""

    while True:
        stdscr.erase()
        stdscr.addstr(0, 0, "Select a recording (Enter to run, q to quit)")
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