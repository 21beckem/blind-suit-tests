import argparse
import time
from dataclasses import dataclass
from pprint import pprint
from threading import Condition, Thread

from serial import Serial


@dataclass(frozen=True)
class Vector3:
	x: float
	y: float
	z: float


@dataclass(frozen=True)
class IMUData:
	acceleration: Vector3
	gyro: Vector3


_latest = None
_latest_timestamp = None
_latest_sequence = 0
_condition = Condition()
_reader_thread = None


def _parse_line(line):
	try:
		values = [float(value.strip()) for value in line.split(",")]
	except ValueError:
		return None

	if len(values) != 6:
		return None

	return IMUData(
		acceleration=Vector3(*values[:3]),
		gyro=Vector3(*values[3:]),
	)


def _read_serial(port, baud):
	global _latest, _latest_timestamp, _latest_sequence

	with Serial(port, baud, timeout=None) as connection:
		while True:
			line = connection.readline().decode("ascii", errors="ignore").strip()
			data = _parse_line(line)
			if data is not None:
				with _condition:
					_latest = data
					_latest_timestamp = time.monotonic()
					_latest_sequence += 1
					_condition.notify_all()


def _start_reader(port, baud):
	global _reader_thread

	if _reader_thread is None:
		_reader_thread = Thread(target=_read_serial, args=(port, baud), daemon=True)
		_reader_thread.start()


def read(port="COM6", baud=115200):
	"""Start the reader and return the most recent six-value IMU sample."""
	_start_reader(port, baud)

	with _condition:
		return _latest


def wait_for_update(port="COM6", baud=115200, timeout=None):
	"""Wait for and return the next IMU sample, or None if timed out."""
	_start_reader(port, baud)

	with _condition:
		sequence = _latest_sequence
		end_time = None if timeout is None else time.monotonic() + timeout
		while _latest_sequence == sequence:
			remaining = None if end_time is None else end_time - time.monotonic()
			if remaining is not None and remaining <= 0:
				return None
			_condition.wait(remaining)
		return _latest


def last_update_time():
	"""Return the monotonic time of the most recent valid IMU sample."""
	with _condition:
		return _latest_timestamp


def main():
	parser = argparse.ArgumentParser(description="Print all IMU data received over serial.")
	parser.add_argument("--port", default="COM6", help="Serial port, e.g. COM3 or /dev/ttyUSB0")
	parser.add_argument("--baud", type=int, default=115200)
	args = parser.parse_args()

	while True:
		data = wait_for_update(args.port, args.baud)
		if data is None:
			print(".", end="")
		else:
			pprint([data.acceleration, data.gyro])


if __name__ == "__main__":
	main()
