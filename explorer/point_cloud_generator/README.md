# Point-cloud generator

This package creates deterministic enclosed four-wall rooms, places relationship-aware furniture, and
returns a dense surface scan. Cabinets are placed against walls and chairs are placed around
tables. The scanner is at the origin, `+Y` is up, `+X` is right, the
view direction is `-Z`, and all distances are metres. That is the convention consumed by
`binocular/with-phone/obstacle_scene.py`.

## Library use

```python
from explorer.point_cloud_generator import (
    generate_point_cloud, generate_scan, generate_scene, scan_scene
)

points = generate_point_cloud({"seed": 7, "objects": {"count": 6}})

# Rich result for simulation, display, and detector benchmarking.
scan = generate_scan({"seed": 7})
print(scan.points.shape, scan.object_ids, scan.objects)

# Keep one room and scan it again from another starting pose. Points returned by
# each scan are expressed in that scanner's local detector frame.
scene = generate_scene({"seed": 7})
first = scan_scene(scene)
second = scan_scene(
    scene,
    sensor_position_m=(1.0, 0.0, -2.0),
    sensor_yaw_deg=90.0,
    omniscient_scan=False,
)
```

`generate_point_cloud()` returns a contiguous `float64` NumPy array shaped `(N, 3)`.
`generate_scan()` additionally returns colors, surface normals, per-point ground-truth object
IDs, semantic classes, object poses, sensor metadata, and the fully resolved configuration.
Scene object poses remain in the stable world frame; points are transformed to the selected
scanner-local frame on each scan.

## Command line

The default command generates a room and opens an interactive Open3D viewer:

```powershell
python -m explorer.point_cloud_generator
python -m explorer.point_cloud_generator --seed 12 --objects 10
python -m explorer.point_cloud_generator --config explorer/point_cloud_generator/default.json
python -m explorer.point_cloud_generator --single-view-scan --sensor-position 1 0 -2 --sensor-yaw 90
```

Press `C` to switch between natural and object-ID colors. Use `--no-view` for automation.
PLY is convenient for general 3D tools;
NPZ preserves every array and embeds the metadata as JSON.

```powershell
python -m explorer.point_cloud_generator --no-view --save-ply room.ply
python -m explorer.point_cloud_generator --no-view --save-npz room.npz --save-metadata room.json
```

Configuration is applied in this order: built-in defaults, JSON file, then CLI overrides.
`default.json` is the single source for package defaults; the Python code does not duplicate
those values. All scan-quality values are tuned directly there.

## Scan modes and flat planes

`omniscient_scan` is controlled by `default.json`. When `true`, it samples every surface in
the room, representing a complete scan made by walking around the entire space. When `false`
(or with `--single-view-scan`), it uses one scanner pose, line-of-sight occlusion, range
limits, and a forward camera frustum. `single_view_horizontal_fov_deg` and
`single_view_vertical_fov_deg` tune that view; floor and wall points inside the frustum are
kept normally.

Large room planes and sufficiently large object faces (including cabinet sides and table
tops) use the same realism model. Their centres receive more dropout and normal-direction
distortion, while plane edges and areas close to other actual surfaces remain clearer:

- `flat_plane_degradation`: enables the effect.
- `flat_plane_min_area_m2`: minimum object-face area that receives the effect; room floor and walls always qualify.
- `flat_plane_dropout`: maximum probability of losing a point in an exposed plane centre.
- `flat_plane_center_noise_m`: distortion at the least-supported part of a plane.
- `flat_plane_edge_noise_m`: distortion at a plane edge or beside an object.
- `flat_plane_edge_width_m`: distance over which edge clarity fades.
- `flat_plane_object_radius_m`: distance over which nearby furniture improves clarity.
- `flat_plane_clarity_power`: shapes how quickly the clean region transitions to the centre.

These effects are additional to the global `noise_sigma_m` and `dropout` settings. Disable
them for ideal geometry with `--no-flat-plane-degradation`.

## Detector integration

```python
import open3d as o3d
from explorer.point_cloud_generator import generate_scan

scan = generate_scan({"seed": 3})
result = obstacle_scene.analyze_point_cloud(
    o3d,
    scan.points,
    obstacle_scene.default_config(),
)
```

Ground-truth IDs are separate from the point matrix. For an ideal segmentation benchmark,
the detector's optional callback can use `scan.object_ids[candidate_mask]`.

