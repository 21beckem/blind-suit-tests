"""Generate a printable calibration checkerboard at an exact physical size.

    python make_board.py                       # 10x7 squares (9x6 inner corners), 25 mm squares, US Letter
    python make_board.py --paper a4
    python make_board.py --square-mm 20

Print at 100% / "Actual size" (NOT "fit to page"), then measure one square with a ruler and enter the real
size with:  python 02_calibrate.py --square-mm 24.8
Glue the sheet flat onto something rigid (cardboard, foam board, a clipboard). A bent board ruins calibration.
"""
import argparse
import struct
import zlib

import cv2
import numpy as np

PAPER_MM = {"letter": (279.4, 215.9), "a4": (297.0, 210.0)}  # landscape (w, h)
DPI = 300


def png_with_dpi(img, dpi):
    ok, buf = cv2.imencode(".png", img)
    data = buf.tobytes()
    ppm = int(round(dpi / 0.0254))
    body = b"pHYs" + struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", 9) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    return data[:33] + chunk + data[33:]          # insert right after the 25-byte IHDR chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=10, help="squares across (inner corners = cols-1)")
    ap.add_argument("--rows", type=int, default=7, help="squares down (inner corners = rows-1)")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--paper", choices=list(PAPER_MM), default="letter")
    args = ap.parse_args()

    px_per_mm = DPI / 25.4
    pw, ph = (int(round(v * px_per_mm)) for v in PAPER_MM[args.paper])
    sq = args.square_mm * px_per_mm
    bw, bh = args.cols * sq, args.rows * sq
    if bw > pw - 20 or bh > ph - 20:
        raise SystemExit("Board %.0fx%.0f mm does not fit on %s paper; use a smaller --square-mm." % (
            args.cols * args.square_mm, args.rows * args.square_mm, args.paper))
    img = np.full((ph, pw), 255, np.uint8)
    x0, y0 = (pw - bw) / 2.0, (ph - bh) / 2.0
    for r in range(args.rows):
        for c in range(args.cols):
            if (r + c) % 2 == 0:
                xa, ya = int(round(x0 + c * sq)), int(round(y0 + r * sq))
                xb, yb = int(round(x0 + (c + 1) * sq)), int(round(y0 + (r + 1) * sq))
                img[ya:yb, xa:xb] = 0
    name = "checkerboard_%dx%d_%gmm_%s.png" % (args.cols, args.rows, args.square_mm, args.paper)
    with open(name, "wb") as f:
        f.write(png_with_dpi(img, DPI))
    print("Wrote %s  (%d x %d inner corners, %g mm squares)" % (name, args.cols - 1, args.rows - 1, args.square_mm))
    print("Print at 100% scale, then measure a square with a ruler. Default calibration expects 9x6 inner corners.")


if __name__ == "__main__":
    main()
