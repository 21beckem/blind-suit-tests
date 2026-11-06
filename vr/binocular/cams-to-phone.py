"""Capture calibrated binocular stereo depth and stream it to the phone."""

import base64
import hashlib
import json
import socket
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from grid import generate_grid
from config import OUTPUT_WIDTH, OUTPUT_HEIGHT, DEPTH_NEAR_METERS, DEPTH_FAR_METERS


# Import the tested binocular camera/calibration implementation.
REPO_ROOT = Path(__file__).resolve().parents[2]
STEREO_DIR = REPO_ROOT / "binocular" / "3d-scan-tests"
sys.path.insert(0, str(STEREO_DIR))

from common import Calibration, Rectifier, StereoCamera, StereoMatcher, load_config  # noqa: E402


HOST = "127.0.0.1"
PORT = 5000

DISPLAY_DEPTH_ALONGSIDE_CAMERA = True
DISPLAY_AT_INPUT_RESOLUTION = True
ROTATE_CAMERA_180 = True


def receive_exactly(conn, size):
    data = bytearray()
    while len(data) < size:
        packet = conn.recv(size - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def read_websocket_frame(conn):
    header = receive_exactly(conn, 2)
    if header is None:
        return None

    first, second = header
    opcode = first & 0x0F
    length = second & 0x7F
    if length == 126:
        extended_length = receive_exactly(conn, 2)
        if extended_length is None:
            return None
        length = int.from_bytes(extended_length, "big")
    elif length == 127:
        extended_length = receive_exactly(conn, 8)
        if extended_length is None:
            return None
        length = int.from_bytes(extended_length, "big")

    # Client-to-server WebSocket frames are masked.
    mask = receive_exactly(conn, 4)
    encoded_payload = receive_exactly(conn, length)
    if mask is None or encoded_payload is None:
        return None
    payload = bytearray(encoded_payload)
    for index in range(len(payload)):
        payload[index] ^= mask[index % 4]
    return opcode, bytes(payload)


def accept_websocket(conn):
    request = b""
    while b"\r\n\r\n" not in request:
        packet = conn.recv(4096)
        if not packet:
            raise ConnectionError("WebSocket handshake closed by phone")
        request += packet

    headers = dict(
        line.split(":", 1)
        for line in request.decode("latin1").split("\r\n")[1:]
        if ":" in line
    )
    key = headers["Sec-WebSocket-Key"].strip()
    accept = base64.b64encode(
        hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
        ).digest()
    ).decode()
    response = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )
    conn.sendall(response.encode())


def send_websocket_text(conn, message):
    payload = message.encode("utf-8")
    length = len(payload)
    if length < 126:
        header = bytes([0x81, length])
    elif length < 65536:
        header = bytes([0x81, 126]) + length.to_bytes(2, "big")
    else:
        header = bytes([0x81, 127]) + length.to_bytes(8, "big")
    conn.sendall(header + payload)


def monitor_phone(conn, disconnected):
    """Read client control frames so a phone disconnect is noticed promptly."""
    try:
        while True:
            frame = read_websocket_frame(conn)
            if frame is None or frame[0] == 8:
                break
    except (ConnectionError, OSError):
        pass
    finally:
        disconnected.set()


def main():
    cfg = load_config()
    stereo_cfg = dict(cfg["stereo"])
    # Keep the configured mapping endpoints measurable even if config.json has a
    # narrower matcher range (the checked-in config currently starts at 0.30 m).
    stereo_cfg["min_depth"] = min(float(stereo_cfg["min_depth"]), DEPTH_NEAR_METERS)
    stereo_cfg["max_depth"] = max(float(stereo_cfg["max_depth"]), DEPTH_FAR_METERS)

    calib = Calibration()
    rect = Rectifier(calib, stereo_cfg["scale"])
    matcher = StereoMatcher(stereo_cfg, rect.fx, rect.baseline)
    print(
        "Stereo depth: %dx%d, baseline %.1f mm, grid %dx%d, %.2f m -> 1, %.2f m -> 0"
        % (
            rect.out_size[0],
            rect.out_size[1],
            rect.baseline * 1000,
            OUTPUT_WIDTH,
            OUTPUT_HEIGHT,
            DEPTH_NEAR_METERS,
            DEPTH_FAR_METERS,
        ),
        flush=True,
    )

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"Waiting for phone WebSocket on {HOST}:{PORT}...", flush=True)

    conn = None
    cam = None
    try:
        conn, addr = server_socket.accept()
        accept_websocket(conn)
        print(f"Connected to phone via: {addr}", flush=True)
        disconnected = threading.Event()
        threading.Thread(target=monitor_phone, args=(conn, disconnected), daemon=True).start()

        cam = StereoCamera(cfg["camera"]).start()
        last_id = -1
        while not disconnected.is_set():
            item = cam.read(last_id, timeout=1.0)
            if item is None:
                continue
            left, right, _timestamp, last_id = item
            rectified_left, rectified_right = rect.rectify(left, right)
            _disparity, depth = matcher.compute(rectified_left, rectified_right)

            # StereoSGBM cannot search for matches in the leftmost disparity
            # range. Crop the same margin from the image and depth so the
            # preview and phone grid cover the same field of view.
            margin = min(matcher.nd, max(depth.shape[1] - 1, 0))
            depth = depth[:, margin:]
            display_left = rectified_left[:, margin:]
            if ROTATE_CAMERA_180:
                # Calibration and positive disparity require the native sensor
                # orientation. Rotate their aligned results for upside-down use.
                depth = cv2.rotate(depth, cv2.ROTATE_180)
                display_left = cv2.rotate(display_left, cv2.ROTATE_180)
            grid = generate_grid(depth)
            send_websocket_text(conn, json.dumps(grid.tolist(), separators=(",", ":")))

            if DISPLAY_DEPTH_ALONGSIDE_CAMERA:
                display_depth = cv2.applyColorMap(
                    np.round(grid * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
                )
                display_left = cv2.cvtColor(display_left, cv2.COLOR_GRAY2BGR)
                display_depth = cv2.resize(
                    display_depth,
                    (display_left.shape[1], display_left.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                if not DISPLAY_AT_INPUT_RESOLUTION:
                    display_left = cv2.resize(display_left, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                    display_depth = cv2.resize(display_depth, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                display = np.hstack((display_left, display_depth))
                cv2.imshow("Binocular Depth -> Phone", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        if cam is not None:
            cam.stop()
        if conn is not None:
            conn.close()
        server_socket.close()
        cv2.destroyAllWindows()
        print("Stream closed.", flush=True)


if __name__ == "__main__":
    main()
