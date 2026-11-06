"""The locally hosted page must load its shared renderer without internet."""

import sys
import threading
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phone_server import PhonePageHandler
from config import V_ON, V_OFF


class PhoneServerTests(unittest.TestCase):
    def test_page_and_renderer_are_served(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), PhonePageHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urlopen(base + "/", timeout=2) as response:
                self.assertTrue(response.geturl().endswith("/navigate-binocular/"))
                page = response.read().decode()
            with urlopen(base + "/vr.js", timeout=2) as response:
                renderer = response.read().decode()
            with urlopen(base + "/hud-values.js", timeout=2) as response:
                values_script = response.read().decode()
            self.assertIn("location.hostname", page)
            self.assertIn("../vr.js", page)
            self.assertIn("../hud-values.js", page)
            self.assertIn("paintHUD", renderer)
            self.assertIn("Math.round(t[i] * 255)", renderer)
            self.assertIn(f'"on": {V_ON}', values_script)
            self.assertIn(f'"off": {V_OFF}', values_script)
            with patch("phone_server.V_ON", 0.3), patch("phone_server.V_OFF", 0.1):
                with urlopen(base + "/hud-values.js", timeout=2) as response:
                    dim_values = response.read().decode()
            self.assertIn('"on": 0.3', dim_values)
            self.assertIn('"off": 0.1', dim_values)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
