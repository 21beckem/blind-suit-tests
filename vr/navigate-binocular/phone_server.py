"""Serve the binocular phone page and its shared VR renderer on the local network."""

import socket
import threading
import ipaddress
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


VR_DIR = Path(__file__).resolve().parents[1]
HTTP_PORT = 8000


class PhonePageHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VR_DIR), **kwargs)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/binocular/")
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
