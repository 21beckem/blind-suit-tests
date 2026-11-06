"""Capture calibrated binocular stereo depth and stream it to the phone."""

import base64
import hashlib
import json
import socket
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from grid import generate_grid
from navigation import Navigator, fuse_maps
from stereo_obstacles import StereoObstacles
from debug_view import render_dashboard
from tuning_controls import NavigationControls, load_settings, save_settings
from phone_server import HTTP_PORT, local_addresses, start_page_server
from config import (
    OUTPUT_WIDTH, OUTPUT_HEIGHT, DEPTH_NEAR_METERS, MAX_AHEAD_METERS,
)


# Import the tested binocular camera/calibration implementation.
REPO_ROOT = Path(__file__).resolve().parents[2]
STEREO_DIR = REPO_ROOT / "binocular" / "3d-scan-tests"
sys.path.insert(0, str(STEREO_DIR))

from common import Calibration, Rectifier, StereoCamera, StereoMatcher, load_config  # noqa: E402


HOST = "0.0.0.0"
PORT = 5000

DISPLAY_DEPTH_ALONGSIDE_CAMERA = True
ROTATE_CAMERA_180 = True


def make_navigator(settings):
    return Navigator(
        path_width=settings.path_width,
        stop_distance=settings.stop_distance,
        near_limit=settings.near_distance,
        ahead_limit=settings.ahead_distance,
        min_observed_fraction=settings.min_coverage,
        switch_frames=settings.switch_frames,
    )


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
    stereo_cfg["max_depth"] = max(float(stereo_cfg["max_depth"]), MAX_AHEAD_METERS)

    calib = Calibration()
    rect = Rectifier(calib, stereo_cfg["scale"])
    matcher = StereoMatcher(stereo_cfg, rect.fx, rect.baseline)
    print(
        "Stereo depth: %dx%d, baseline %.1f mm, navigation HUD %dx%d, range %.2f-%.2f m"
        % (
            rect.out_size[0],
            rect.out_size[1],
            rect.baseline * 1000,
            OUTPUT_WIDTH,
            OUTPUT_HEIGHT,
            DEPTH_NEAR_METERS,
            MAX_AHEAD_METERS,
        ),
        flush=True,
    )

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    page_server = None

    conn = None
    cam = None
    keep_serving = True
    try:
        page_server = start_page_server()
        addresses = local_addresses()
        for address in addresses:
            print(f"Open http://{address}:{HTTP_PORT}/ on your phone", flush=True)
        if not addresses:
            print(f"Open http://<computer-LAN-IP>:{HTTP_PORT}/ on your phone", flush=True)
        print(f"Waiting for phone WebSocket on port {PORT}...", flush=True)
        conn, addr = server_socket.accept()
        accept_websocket(conn)
        print(f"Connected to phone via: {addr}", flush=True)
        disconnected = threading.Event()
        threading.Thread(target=monitor_phone, args=(conn, disconnected), daemon=True).start()

        cam = StereoCamera(cfg["camera"]).start()
        settings = load_settings()
        navigator = make_navigator(settings)
        controls = NavigationControls(settings) if DISPLAY_DEPTH_ALONGSIDE_CAMERA else None
        observer = None
        last_id = -1
        last_send_at = None
        while not disconnected.is_set():
            item = cam.read(last_id, timeout=1.0)
            if item is None:
                continue
            if controls is not None:
                updated = controls.read()
                if updated != settings:
                    settings = updated
                    navigator = make_navigator(settings)
            left, right, capture_timestamp, last_id = item
            frame_start = time.perf_counter()
            rectified_left, rectified_right = rect.rectify(left, right)
            _disparity, depth = matcher.compute(rectified_left, rectified_right)
            stereo_ms = (time.perf_counter() - frame_start) * 1000

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
            if observer is None:
                cx = rect.cx - margin
                cy = rect.cy
                if ROTATE_CAMERA_180:
                    cx = depth.shape[1] - 1 - cx
                    cy = depth.shape[0] - 1 - cy
                observer = StereoObstacles(
                    fx=rect.fx, fy=rect.fy, cx=cx, cy=cy,
                    width=depth.shape[1], height=depth.shape[0],
                    near=DEPTH_NEAR_METERS, far=MAX_AHEAD_METERS,
                    min_hazard_points=settings.hazard_points,
                )
            observer.min_hazard_points = settings.hazard_points
            observe_start = time.perf_counter()
            scene = fuse_maps([observer.observe(depth)])
            decision = navigator.decide(scene)
            grid = generate_grid(decision)
            observe_ms = (time.perf_counter() - observe_start) * 1000
            send_websocket_text(conn, json.dumps(grid.tolist(), separators=(",", ":")))
            sent_at = time.monotonic()
            message_gap_ms = 0 if last_send_at is None else (sent_at - last_send_at) * 1000
            last_send_at = sent_at

            if DISPLAY_DEPTH_ALONGSIDE_CAMERA:
                stats = {
                    "camera_fps": cam.fps,
                    "frame_age_ms": max(0, (time.time() - capture_timestamp) * 1000),
                    "stereo_ms": stereo_ms,
                    "observe_ms": observe_ms,
                    "total_ms": (time.perf_counter() - frame_start) * 1000,
                    "message_gap_ms": message_gap_ms,
                }
                display = render_dashboard(
                    display_left, depth, scene, decision, grid, observer.debug, stats,
                    near=DEPTH_NEAR_METERS, far=MAX_AHEAD_METERS,
                    settings=settings,
                )
                cv2.imshow("Binocular Navigation Debug", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("r"):
                    capture_dir = Path(__file__).resolve().parent / "captures"
                    capture_dir.mkdir(exist_ok=True)
                    capture_path = capture_dir / f"depth-{time.strftime('%Y%m%d-%H%M%S')}-{last_id}.npz"
                    np.savez_compressed(
                        capture_path, depth=depth, fx=observer.fx, fy=observer.fy,
                        cx=observer.cx, cy=observer.cy,
                        settings=json.dumps(asdict(settings)),
                    )
                    print(f"Saved depth sample: {capture_path}", flush=True)
                if key == ord("s"):
                    save_settings(settings)
                    print("Saved navigation settings", flush=True)
                if key == ord("d") and controls is not None:
                    settings = controls.reset()
                    navigator = make_navigator(settings)
                    observer.min_hazard_points = settings.hazard_points
                if key in (ord("q"), 27):
                    keep_serving = False
                    break
    finally:
        if cam is not None:
            cam.stop()
        if conn is not None:
            conn.close()
        server_socket.close()
        if page_server is not None:
            page_server.shutdown()
            page_server.server_close()
        cv2.destroyAllWindows()
        print("Stream closed.", flush=True)
    return keep_serving


if __name__ == "__main__":
    while main():
        print("Phone disconnected; restarting local server...", flush=True)
