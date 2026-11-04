"""Run this first: `python selftest.py`. Needs NO camera. Verifies your install and the whole pipeline
math on synthetic data (calibration, stereo depth, tracking, TSDF fusion, saving)."""
import sys
import tempfile
import traceback

import numpy as np

results = []


def check(name, fn):
    try:
        msg = fn()
        print("  PASS  %-28s %s" % (name, msg or ""))
        results.append(True)
    except Exception as e:  # noqa: BLE001
        print("  FAIL  %-28s %s" % (name, e))
        traceback.print_exc()
        results.append(False)


def t_imports():
    import cv2
    import open3d as o3d
    return "numpy %s, opencv %s, open3d %s" % (np.__version__, cv2.__version__, o3d.__version__)


def t_calibration():
    import cv2
    import calib_tools as ct
    rng = np.random.default_rng(1)
    size = (1280, 800)
    K = np.array([[700.0, 0, 640], [0, 700.0, 400], [0, 0, 1]])
    D = np.array([-0.02, 0.01, 0, 0, 0.0])
    Rs, _ = cv2.Rodrigues(np.array([0.0, 0.004, 0.001]))
    Ts = np.array([[-0.06], [0.0008], [0.0004]])
    cols, rows, sq = 9, 6, 0.025
    obj = ct.board_object_points(cols, rows, sq)
    objs, pl, pr = [], [], []
    while len(objs) < 25:
        rv = rng.uniform(-0.5, 0.5, 3)
        tv = np.array([rng.uniform(-0.1, 0.05), rng.uniform(-0.08, 0.05), rng.uniform(0.5, 1.0)])
        l, _ = cv2.projectPoints(obj, rv, tv, K, D)
        rvr, tvr = cv2.composeRT(rv.reshape(3, 1), tv.reshape(3, 1), cv2.Rodrigues(Rs)[0].reshape(3, 1), Ts.reshape(3, 1))[:2]
        r, _ = cv2.projectPoints(obj, rvr, tvr, K, D)
        if not (l[:, 0, 0].min() > 5 and l[:, 0, 0].max() < 1275 and l[:, 0, 1].min() > 5 and l[:, 0, 1].max() < 795
                and r[:, 0, 0].min() > 5 and r[:, 0, 0].max() < 1275):
            continue
        objs.append(obj)
        pl.append((l + rng.normal(0, 0.15, l.shape)).astype(np.float32))
        pr.append((r + rng.normal(0, 0.15, r.shape)).astype(np.float32))
    res = ct.run_calibration(objs, pl, pr, size, log=lambda *_: None)
    rect = ct.build_rectification(res, size)
    mean_e, _ = ct.rectified_epipolar_error(res, rect, pl, pr)
    base_mm = rect["baseline"] * 1000
    assert abs(base_mm - 60.0) < 1.5, "baseline %.2f mm (expected ~60)" % base_mm
    assert mean_e < 0.3, "epipolar error %.3f px" % mean_e
    return "baseline %.2f mm (true 60), epipolar err %.3f px, RMS %.3f" % (base_mm, mean_e, res["rms"])


def _texture(w, h, seed=3):
    import cv2
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 255, (h, w)).astype(np.uint8)
    t = cv2.GaussianBlur(t, (0, 0), 1.6)
    return cv2.normalize(t, None, 0, 255, cv2.NORM_MINMAX)


def t_stereo_depth():
    import cv2
    from common import DEFAULTS, StereoMatcher
    fx, B, z = 360.0, 0.06, 2.0
    d = fx * B / z
    left = _texture(640, 400)
    right = cv2.warpAffine(left, np.float32([[1, 0, -d], [0, 1, 0]]), (640, 400), borderMode=cv2.BORDER_REPLICATE)
    m = StereoMatcher(DEFAULTS["stereo"], fx, B)
    disp, depth = m.compute(left, right)
    roi = depth[50:350, 150:600]
    v = roi[roi > 0]
    assert v.size > 0.5 * roi.size, "only %.0f%% valid" % (100.0 * v.size / roi.size)
    med = float(np.median(v))
    assert abs(med - z) < 0.06, "median depth %.3f m, expected %.2f" % (med, z)
    return "median depth %.3f m (true %.2f), %.0f%% valid, closest measurable %.2f m" % (
        med, z, 100.0 * v.size / roi.size, m.closest_depth())


def _session(fusion=None):
    from common import DEFAULTS
    from scanner_core import ScanSession
    return ScanSession(360.0, 360.0, 319.5, 199.5, 640, 400, DEFAULTS["scan"], fusion=fusion)


def t_tracking():
    import cv2
    s = _session()
    z, fx = 2.0, 360.0
    tex = _texture(900, 400, seed=5)
    depth = np.full((400, 640), z, np.float32)
    poses = []
    for tx in (0.0, 0.03, 0.06, 0.09):
        shift = -fx * tx / z      # camera moves +x -> image content moves -x
        img = cv2.warpAffine(tex, np.float32([[1, 0, 130 + shift], [0, 1, 0]]), (640, 400),
                             borderMode=cv2.BORDER_REPLICATE)
        info = s.process(img, depth)
        poses.append(s.pose[0, 3])
        assert info["state"] == "TRACKING", "state %s at tx=%.2f" % (info["state"], tx)
    err = abs(poses[-1] - 0.09)
    assert err < 0.012, "final x %.4f m, expected 0.09" % poses[-1]
    return "tracked +9.0 cm sideways -> estimated %.1f cm, %d keyframes" % (poses[-1] * 100, s.n_integrated)


def t_tracking_stereo_depth():
    """Regression: tracking must also work on NOISY stereo-derived depth (not just perfect depth)."""
    import cv2
    from common import DEFAULTS, StereoMatcher
    fx, B, z = 350.0, 0.06, 2.0
    tex = _texture(1300, 400, seed=2)
    m = StereoMatcher(DEFAULTS["stereo"], fx, B)
    s = _session()
    d = fx * B / z
    for i in range(7):
        tx = 0.01 * i
        sh = 65 - fx * tx / 2.0
        L = cv2.warpAffine(tex, np.float32([[1, 0, sh], [0, 1, 0]]), (640, 400), borderMode=cv2.BORDER_REPLICATE)
        R = cv2.warpAffine(tex, np.float32([[1, 0, sh - d], [0, 1, 0]]), (640, 400), borderMode=cv2.BORDER_REPLICATE)
        s.process(L, m.compute(L, R)[1])
    err = abs(s.pose[0, 3] - 0.06)
    assert s.state == "TRACKING" and err < 0.015, "state %s, x=%.3f m (true 0.060)" % (s.state, s.pose[0, 3])
    return "tracked +6.0 cm on stereo depth -> %.1f cm" % (s.pose[0, 3] * 100)


def _fusion_case(fusion):
    s = _session(fusion)
    depth = np.full((400, 640), 1.513, np.float32)  # avoid sitting exactly on a voxel boundary
    yy, xx = np.mgrid[0:400, 0:640]
    depth -= (0.3 * np.exp(-(((xx - 320) / 90.0) ** 2 + ((yy - 200) / 90.0) ** 2))).astype(np.float32)
    s.process(_texture(640, 400, seed=9), depth)
    pts, cols = s.extract_cloud()
    assert len(pts) > 1000, "only %d points extracted" % len(pts)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        base = s.save(td, log=lambda *_: None)
        assert base is not None, "save produced nothing"
    return len(pts)


def t_fusion_points():
    n = _fusion_case("points")
    return "point-cloud fusion: %d points, Poisson mesh + save OK" % n


def t_fusion_tsdf():
    from scanner_core import tsdf_works
    if not tsdf_works():
        return "SKIPPED - this Open3D build's TSDF returns nothing; the scanner will auto-use point fusion"
    n = _fusion_case("tsdf")
    return "TSDF fusion: %d points, mesh + save OK" % n


def main():
    print("Stereo scanner self-test\n")
    check("imports", t_imports)
    check("stereo calibration math", t_calibration)
    check("stereo depth (SGBM)", t_stereo_depth)
    check("tracking (RGB-D odometry)", t_tracking)
    check("tracking on stereo depth", t_tracking_stereo_depth)
    check("fusion: point cloud", t_fusion_points)
    check("fusion: TSDF", t_fusion_tsdf)
    ok = all(results)
    print("\n%s" % ("ALL PASSED" if ok else "SOME TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
