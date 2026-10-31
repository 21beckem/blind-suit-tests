import base64
import hashlib
import json
import queue
import socket
import threading
import time

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import pico

HOST = "127.0.0.1"
PORT = 5000
MODEL_NAME = "LiheYoung/depth-anything-small-hf"
PROCESS_IMAGE_SIZE = 252
OUTPUT_WIDTH = 8
OUTPUT_HEIGHT = 8
DISPLAY_DEPTH_ALONGSIDE_CAMERA = True
DISPLAY_AT_INPUT_RESOLUTION = True
PICO_PORT = "COM12"
PICO_BAUD = 115200
PICO_VALID_DISTANCE_MAX_MM = 3500
PICO_MODE2_VALID_FRACTION = 0.01
PICO_DATA_TIMEOUT_SECONDS = 2.0
PICO_COMBINATION_MODE = "blend"  # "fill" or "blend"
PICO_WEIGHT = 0.90
MODE1_MARKER = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)

OUTPUT_ASPECT_RATIO = OUTPUT_WIDTH / OUTPUT_HEIGHT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading {MODEL_NAME} on {device}...", flush=True)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
depth_model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).to(device)
depth_model.eval()
print("Depth model ready.", flush=True)


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


def crop_to_aspect_ratio(image, aspect_ratio):
    height, width = image.shape[:2]
    image_aspect_ratio = width / height
    if image_aspect_ratio > aspect_ratio:
        cropped_width = int(height * aspect_ratio)
        left = (width - cropped_width) // 2
        return image[:, left:left + cropped_width]

    cropped_height = int(width / aspect_ratio)
    top = (height - cropped_height) // 2
    return image[top:top + cropped_height, :]


def normalize_depth(depth):
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max > depth_min:
        return (depth - depth_min) / (depth_max - depth_min)
    return np.zeros_like(depth)


def map_pico_depth(pico_grid):
    """Map the complete 0-3500 mm Pico range to 1.0-0.3."""
    mapped = 1.0 - (
        np.asarray(pico_grid, dtype=np.float32)
        / PICO_VALID_DISTANCE_MAX_MM
    ) * 0.7
    return np.clip(mapped, 0.3, 1.0)


def fuse_depth(depth, pico_grid, pico_timestamp, now=None):
    """Return the fused grid and whether it is in Pico-backed Mode 2."""
    if now is None:
        now = time.monotonic()

    depth = np.asarray(depth, dtype=np.float32)
    is_fresh = (
        pico_grid is not None
        and pico_timestamp is not None
        and now - pico_timestamp <= PICO_DATA_TIMEOUT_SECONDS
    )
    if not is_fresh:
        depth[0] = MODE1_MARKER
        return depth, False

    pico_grid = np.asarray(pico_grid, dtype=np.float32)
    valid = (
        np.isfinite(pico_grid)
        & (pico_grid >= 0)
        & (pico_grid < PICO_VALID_DISTANCE_MAX_MM)
    )
    required_tiles = int(np.ceil(depth.size * PICO_MODE2_VALID_FRACTION))
    if int(valid.sum()) < required_tiles:
        depth[0] = MODE1_MARKER
        return depth, False

    depth_mapped = 0.3 + (depth * 0.7)
    pico_mapped = map_pico_depth(pico_grid)
    if PICO_COMBINATION_MODE == "fill":
        fused = np.where(valid, pico_mapped, depth_mapped)
    elif PICO_COMBINATION_MODE == "blend":
        blended = (
            PICO_WEIGHT * pico_mapped
            + (1.0 - PICO_WEIGHT) * depth_mapped
        )
        fused = np.where(valid, blended, depth_mapped)
    else:
        raise ValueError(
            "PICO_COMBINATION_MODE must be 'fill' or 'blend'"
        )
    return fused.astype(np.float32), True


def receive_frames(conn, frame_queue):
    while True:
        frame = read_websocket_frame(conn)
        if frame is None:
            frame_queue.put(None)
            return
        opcode, jpg_bytes = frame
        if opcode == 8:
            frame_queue.put(None)
            return
        if opcode != 2:
            continue
        try:
            frame_queue.put_nowait(jpg_bytes)
        except queue.Full:
            frame_queue.get_nowait()
            frame_queue.put_nowait(jpg_bytes)


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"Waiting for phone camera WebSocket on {HOST}:{PORT}...", flush=True)
conn, addr = server_socket.accept()
accept_websocket(conn)
print(f"Connected to phone via: {addr}", flush=True)

frame_queue = queue.Queue(maxsize=1)
reader = threading.Thread(
    target=receive_frames, args=(conn, frame_queue), daemon=True
)
reader.start()
# cv2.namedWindow("Live Phone Camera Feed", cv2.WINDOW_NORMAL)

try:
    while True:
        try:
            jpg_bytes = frame_queue.get(timeout=0.1)
        except queue.Empty:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        if jpg_bytes is None:
            break

        image = cv2.imdecode(
            np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            continue

        image = crop_to_aspect_ratio(image, OUTPUT_ASPECT_RATIO)
        cropped_size = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = processor(
            images=rgb_image,
            size={
                "height": PROCESS_IMAGE_SIZE,
                "width": PROCESS_IMAGE_SIZE,
            },
            return_tensors="pt",
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = depth_model(**inputs)
        depth = processor.post_process_depth_estimation(
            outputs, target_sizes=[cropped_size]
        )[0]["predicted_depth"]
        depth = depth.cpu().numpy()
        depth = cv2.resize(
            normalize_depth(depth),
            (OUTPUT_WIDTH, OUTPUT_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        pico_grid = pico.read(PICO_PORT, PICO_BAUD)
        fused_depth, mode2 = fuse_depth(
            depth,
            pico_grid,
            pico.last_update_time(),
        )
        send_websocket_text(
            conn,
            json.dumps(fused_depth.tolist(), separators=(",", ":")),
        )
        depth = cv2.applyColorMap(
            np.round(fused_depth * 255).astype(np.uint8),
            cv2.COLORMAP_INFERNO,
        )
        image = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        depth = cv2.resize(depth, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        if DISPLAY_AT_INPUT_RESOLUTION:
            display_width = cropped_size[1]
            display_height = cropped_size[0]
            image = cv2.resize(
                image, (display_width, display_height), interpolation=cv2.INTER_NEAREST
            )
            depth = cv2.resize(
                depth, (display_width, display_height), interpolation=cv2.INTER_NEAREST
            )
        if DISPLAY_DEPTH_ALONGSIDE_CAMERA:
            display = np.hstack((image, depth))
        else:
            display = depth
        mode_name = "Pico depth" if mode2 else "DepthAnything image"
        cv2.imshow(f"Live Phone Camera Feed - {mode_name}", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
    print("Stream closed.")
