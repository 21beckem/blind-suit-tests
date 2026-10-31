import base64
import hashlib
import socket

import cv2
import numpy as np

HOST = "127.0.0.1"
PORT = 5000


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
        length = int.from_bytes(receive_exactly(conn, 2), "big")
    elif length == 127:
        length = int.from_bytes(receive_exactly(conn, 8), "big")

    mask = receive_exactly(conn, 4)
    payload = bytearray(receive_exactly(conn, length))
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


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"Waiting for phone camera WebSocket on {HOST}:{PORT}...", flush=True)
conn, addr = server_socket.accept()
accept_websocket(conn)
print(f"Connected to phone via: {addr}", flush=True)

cv2.namedWindow("Live Phone Camera Feed", cv2.WINDOW_NORMAL)

try:
    while True:
        frame = read_websocket_frame(conn)
        if frame is None:
            break
        opcode, jpg_bytes = frame
        if opcode == 8:
            break
        if opcode != 2:
            continue

        image = cv2.imdecode(
            np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is not None:
            cv2.imshow("Live Phone Camera Feed", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
    print("Stream closed.")
