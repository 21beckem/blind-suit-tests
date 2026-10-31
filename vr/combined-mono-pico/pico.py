import argparse
import time
from pprint import pprint
from threading import Lock, Thread

from serial import Serial

_connection = None
_buffer = b""
_rows = {}
_latest = None
_latest_timestamp = None
_lock = Lock()
_reader_thread = None


def _read_serial(port, baud):
	global _connection, _buffer, _rows, _latest, _latest_timestamp

	_connection = Serial(port, baud, timeout=None)
	while True:
		_buffer += _connection.read(_connection.in_waiting or 1)
		while b"\n" in _buffer:
			line, _buffer = _buffer.split(b"\n", 1)
			line = line.decode("ascii", errors="ignore").strip()
			if not line.startswith("y") or ":" not in line:
				continue

			row_name, values = line.split(":", 1)
			try:
				row_index = int(row_name[1:])
				row = [int(value.strip()) for value in values.split(",") if value.strip()]
			except ValueError:
				continue

			if 0 <= row_index < 8 and len(row) == 8:
				_rows[row_index] = row
			if len(_rows) == 8:
				with _lock:
					_latest = [_rows[index][:] for index in range(8)]
					_latest_timestamp = time.monotonic()
				_rows.clear()


def read(port="COM12", baud=115200):
	"""Start the background reader and return the most recent complete array."""
	global _reader_thread

	if _reader_thread is None:
		_reader_thread = Thread(target=_read_serial, args=(port, baud), daemon=True)
		_reader_thread.start()

	with _lock:
		return None if _latest is None else [row[:] for row in _latest]


def last_update_time():
	"""Return the monotonic time of the most recent complete array."""
	with _lock:
		return _latest_timestamp


def main():
	parser = argparse.ArgumentParser(description="Print all data received over serial.")
	parser.add_argument("--port", default="COM12", help="Serial port, e.g. COM3 or /dev/ttyUSB0")
	parser.add_argument("--baud", type=int, default=115200)
	args = parser.parse_args()

	while True:
		pprint(read(args.port, args.baud))


if __name__ == "__main__":
	main()
