"""Serve the binocular phone page and its shared VR renderer on the local network."""

import socket
import threading
import ipaddress
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hud_protocol import hud_config


VR_DIR = Path(__file__).resolve().parents[1]
HTTP_PORT = 8000


class PhonePageHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VR_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/hud-values.js":
            payload = ("window.HUD_VALUES = Object.freeze(" +
                       json.dumps(hud_config()) + ");\n").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/navigate-binocular/")
            self.end_headers()
            return
        super().do_GET()


def local_addresses():
    """Likely LAN addresses; the phone and computer must share a local network."""
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(
            socket.gethostname(), None, family=socket.AF_INET)}
    except OSError:
        addresses = set()
    return sorted(address for address in addresses if
                  ipaddress.ip_address(address).is_private
                  and not ipaddress.ip_address(address).is_link_local
                  and not ipaddress.ip_address(address).is_loopback)


def start_page_server(port=HTTP_PORT):
    server = ThreadingHTTPServer(("0.0.0.0", port), PhonePageHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    server = start_page_server()
    addresses = local_addresses()
    for address in addresses:
        print(f"Open http://{address}:{HTTP_PORT}/ on your phone", flush=True)
    if not addresses:
        print(f"Open http://<computer-LAN-IP>:{HTTP_PORT}/ on your phone", flush=True)
    print("Page server running; press Ctrl+C to stop.", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
