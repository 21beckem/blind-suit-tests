import base64
import hashlib
import queue
import socket
import threading

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

HOST = "127.0.0.1"
PORT = 5000
MODEL_NAME = "LiheYoung/depth-anything-small-hf"
PROCESS_IMAGE_SIZE = 252
OUTPUT_WIDTH = 8
OUTPUT_HEIGHT = 8
DISPLAY_DEPTH_ALONGSIDE_CAMERA = True
DISPLAY_AT_INPUT_RESOLUTION = True

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
cv2.namedWindow("Live Phone Camera Feed", cv2.WINDOW_NORMAL)

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
        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
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
        cv2.imshow("Live Phone Camera Feed - Depth Anything", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
    print("Stream closed.")
