"""The locally hosted page must load its shared renderer without internet."""

import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phone_server import PhonePageHandler


class PhoneServerTests(unittest.TestCase):
    def test_page_and_renderer_are_served(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), PhonePageHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urlopen(base + "/", timeout=2) as response:
                page = response.read().decode()
            with urlopen(base + "/vr.js", timeout=2) as response:
                renderer = response.read().decode()
            self.assertIn("location.hostname", page)
            self.assertIn("../vr.js", page)
            self.assertIn("paintHUD", renderer)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
