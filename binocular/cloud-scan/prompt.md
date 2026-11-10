# Guide: autonomously build a live 3D obstacle map from the stereo depth stream



Audience: the AI implementing this. Author: a second AI that has read none of the code but knows the

depth pipeline's numbers and the usual pitfalls. **Read the existing modules first and adapt this plan

to them. Do not change the `CameraFeed` / `DepthMapper` / `DepthWindow` APIs; add new modules beside them.**



## 0. Operating constraints and autonomous workflow

The only currently available recording is an apartment walk-through. Treat it as the initial
acceptance fixture: inspect it, determine its actual contents and timing from the repository, and
build the first complete working implementation against it. Do not require the human to record,
measure, label, review, tune, configure, or answer questions.

The implementation must:

- Discover the existing code, calibration, recordings, and test commands on its own.
- Make reasonable defaults from observed repository evidence and record them in code/configuration.
- Generate synthetic scenes and deterministic fixtures for geometry and transform tests instead of
  requesting a human-controlled scene.
- Run the available recording end to end offline, save machine-readable metrics and debug images,
  and use those results to reject regressions.
- Be robust to additional recordings appearing later: discover all compatible recordings
  automatically, process each independently, and report when a recording is unsuitable rather than
  failing silently.
- Complete the first full working draft before treating extra recordings as useful validation.

When a requirement cannot be measured from the apartment recording, use a synthetic test or an
explicitly reported limitation. Never turn that limitation into a manual task.

## 0. What we are building (and what we are not)



A person wears a side-by-side USB stereo camera (one 1280x480 frame = left | right) connected to a laptop

(CPU only, no GPU). We need a map around them that is **good enough and consistent enough to detect

obstacles**, including things they saw a few seconds ago and have since turned away from.



- **Not** a pretty, dense, globally consistent 3D scan. No loop closure, no mesh, no photoreal colour.

- **Yes** a rolling map of roughly 6 m x 6 m around the person that says per cell: free / obstacle / unknown,

  with the height of what is there.

- Must be **live** (target under 30 ms for the mapping step at half-resolution depth, on top of depth itself).

- Prefer **false "unknown"** over **false "free"**. Never mark space free that was not actually seen to be free.



## 1. Facts about the sensor (measured on the provided calibration)



- Focal length 739.6 px at 640x480, baseline 6.05 cm, so f*B = 44.7 px*m at full resolution, 22.4 at half.

- At half resolution (`scale=0.5`, the repository's configured setting) the nearest measurable distance is about 0.48 m.

  Anything closer, and everything below the bottom of the frame (the person's feet), is a **blind zone**.

- Depth error grows with the square of distance. Assuming 0.35 px matching noise (measured about 0.3 px

  on flat surfaces at full resolution; unmeasured at half), the 1-sigma depth error at half resolution is

  about 2 cm at 1 m, 6 cm at 2 m, 14 cm at 3 m, 25 cm at 4 m. **Do not trust the map beyond about 3 m.**

  Beyond that, at most report "something tall is out there".

- Depth results come with a confidence: 0 none, 1 weak (low texture, interpolated by the matcher), 2 solid.

  **Use only confidence 2 for geometry.** White tabletops, black cabinets and blown-out windows come back weak or empty.

  They must become "unknown", not "free".

- The camera is not level: it hangs on a person, pitched and rolled, and it bobs with every step.



## 2. Architecture



New modules (names are suggestions), each testable on its own and each driven from the recorded feed:



| Module | Job |

|---|---|

| `ground_plane.py` | Per frame: fit the floor plane; return gravity direction, camera height, tilt, and a validity flag. |

| `pose_tracker.py` | Per frame: estimate camera motion since the previous frame; return a pose and a tracking-quality flag. |

| `obstacle_map.py` | The rolling 2.5D grid: integrate points, decay, query. |

| `map_view.py` (or extend `ui.py`) | Top-down picture of the grid with the camera and its field of view drawn. Needed to debug anything. |

| `config.py` | Add a `map` section with every threshold below. No magic numbers in code. |



Data flow in the main loop, one extra benchmark bucket:



```

left, right = feed.read()

result = depth.compute(left, right)          # existing: disp, conf, gray, points

floor  = ground.fit(result)                  # new

pose   = tracker.update(result, floor)       # new (stage 3; identity before that)

grid.integrate(result, floor, pose)          # new

window.update(result, grid)                  # show depth + top-down map

```



### Representation: a 2.5D height grid, not a voxel cloud

- Fixed **world-frame** grid, gravity aligned (the grid plane is the floor), about 6 m x 6 m at 5 cm cells = 120x120 cells.

  The person is somewhere inside it; when they get within about 1.5 m of an edge, **scroll** the grid by whole

  cells (shift the arrays and clear the newly exposed rows), rather than resampling it every frame. Resampling blurs.

- Per cell store, as separate numpy arrays: `max_height` above the floor, `min_height`, `hits` (obstacle evidence),

  `floor_hits` (evidence of seeing bare floor), `last_seen` (frame time).

- Cell state is derived, not stored: **obstacle** if enough obstacle evidence; **free** if enough floor evidence and

  no obstacle evidence; otherwise **unknown**.

- A full voxel cloud costs more and gives nothing extra for obstacle detection. If a 3D picture is wanted, draw it

  in a separate viewer from decimated points. Never let a viewer slow the mapping loop.

- This is the same idea as ETH's elevation mapping and the occupancy layers in ROS Nav2 / OctoMap. Skim their

  descriptions for terminology (log-odds, per-cell variance), but do not import any of them.



## 3. Build in stages. Do not start a stage until the previous one passes its automated test.



### Stage 0: orient yourself (no new features)

- Read every module and inspect the recording programmatically. Determine what `depth.compute()`
  returns, its units, the frame convention (OpenCV camera frame: x right, y down, z forward,
  metres), the recording contents, frame count, frame rate, resolution, and timestamp behavior.
  Store durable discoveries in the implementation's existing configuration or diagnostics format;
  do not create a manual handoff document as a prerequisite.
- Validate camera/recording compatibility from metadata and decoded frames. Report concrete
  incompatibilities in the automated run and choose a safe fallback when possible.



### Stage 1: single-frame local obstacle map (person stationary, no tracking)

1. **Points**: take confidence-2 pixels, subsample (every 2nd or 3rd pixel is plenty), keep only z between 0.5 and 3.5 m.

2. **Floor plane**: RANSAC plane fit on the lower part of the image and on points between 0.5 and 2.5 m

   (3-4 cm inlier threshold, a few hundred iterations on a subsample, vectorised numpy). Accept the fit only if

   there are enough inliers (suggest at least 10 percent of the sampled points), the normal is within about 40 degrees of the

   previous good normal (or of "camera roughly upright" on the first frame), and the camera height comes out plausible.

   Otherwise **keep the last good plane and mark it stale**; stale for more than about 1 s means the map stops integrating.

   Smooth the accepted plane lightly (exponential average); do not smooth so hard that it lags behind head-bob.

3. **Gravity frame**: build a rotation so the floor normal points up. Define the axes once and write them down

   (suggest x right, y up, z forward, floor at y = 0). Sign mistakes here cause hours of confusion, so write an automated unit test with a

   synthetic scene: a flat floor plus a 0.4 m box at a known position, camera pitched 20 degrees, expect the box in the right cell.

4. **Classify points by height above the floor**:

   - floor: |h| below about 0.05 m

   - obstacle: h between a range-dependent minimum and about 2.0 m. Use `min_h = max(0.08, 2 * sigma_z(z))` so noise at range

     cannot create phantom obstacles. Cap at 2.0 m (the person's head), because a hanging lamp or a high shelf is not an obstacle.

     **Tabletops at 0.7 m are obstacles**.

   - hole / drop-off: points well **below** the floor, or a floor patch that should be visible but is not. Mark such cells "unknown/danger", never "free".

5. **Scatter into the grid** with `np.add.at` or `np.bincount` (vectorised). If it is still too slow, use numba. Do not loop in Python over pixels.

6. **Free space**: floor-inlier points mark their cells as seen-floor. Optional, stage 2: carve the cells between the camera and each hit.



**Automated test (offline, on the available recording and generated fixtures)**: verify map
invariants numerically and emit top-down debug images for diagnostics. A synthetic flat floor
produces essentially no obstacle cells, and synthetic boxes at 1.0, 1.5, 2.0, and 2.5 m land in
the expected cells within about 10 cm at 1-2 m and about 20 cm at 2.5 m. Use any apartment-scene
observations only as supplemental, unlabeled evidence; do not require human ground truth.



### Stage 2: temporal fusion (still stationary)

- Require **persistence**: a cell becomes an obstacle after it is hit in k of the last n frames (suggest 3 of 5). One frame of noise must not create one.

- **Asymmetric update**: obstacle evidence rises quickly and decays slowly. Free evidence rises slowly. A noisy "free" must not erase a real obstacle.

- **Decay**: `last_seen` older than about 10-20 s fades toward unknown. This handles people walking through and moved chairs.

- Optional ray carving: for each image column (or a coarse set of bearings), cells nearer than the nearest hit and inside the field of view become free.

  Keep it conservative: only carve where there is solid depth, and only reduce obstacle evidence a little per frame.



**Automated test**: replay the walking recording and verify that transient observations decay
without a permanent trail. Use generated frame sequences when the recording does not contain a
usable occurrence of the required transition.

Static-scene jitter: obstacle edges move less than about one cell between frames.



### Stage 3: camera motion tracking (the hard part; unlocks the "behind me" memory)

Goal: a pose (x, z, yaw, plus the gravity frame from the floor plane) good for a few seconds and a few metres. **Drift is acceptable because the map is local and decays.**



Recommended approach (cheap and robust enough):

1. Detect corners on the previous left image (`cv2.goodFeaturesToTrack`, bucketed over a grid so they spread across the image), only at pixels with **confidence-2 depth**.

2. Track them into the current left image with `cv2.calcOpticalFlowPyrLK` (pyramid, small window; run on the half-resolution image).

3. Lift the previous-frame points to 3D with the previous depth; solve `cv2.solvePnPRansac` against the current-frame pixels

   (reprojection threshold about 2 px, need at least 25 inliers, and inliers spread over the image, not clustered).

4. Take yaw and horizontal translation from the result; **take pitch, roll and height from the floor plane**, which is much more reliable. Do not let the two fight.

5. Reject implausible motion (more than about 1.5 m/s or 180 deg/s) and fall back on a constant-velocity guess for at most about 0.3 s.

6. **If tracking fails (few inliers, low texture, motion blur): stop integrating into the map** and mark tracking lost.

   A bad pose written into the map creates ghost walls that stay for the whole decay time, and that is the worst failure mode of the entire project.

   Fade confidence while lost; when features return, re-track against the last good keyframe (ORB match + PnP) or just continue and accept a seam.

7. Use **keyframes** (re-anchor when the camera has moved about 0.3 m or turned about 15 degrees) to reduce drift from chaining frame-to-frame.



Moving people are a minority of the features, so RANSAC rejects them. Large white walls, whiteboards and tabletops

give few features: expect lost tracking there and design for it. Do not try to fix it with ICP on the point cloud as the primary method:

flat floors and walls make ICP slide along them and drift badly. (ICP can be a later refinement.)



Alternatives if this is too fragile: yaw-only tracking (rotation from feature matches), which covers the main case

of "the person turned around"; or an IMU if the camera module has one (check; most do not).



**Automated test**: use the apartment walk-through to measure tracking continuity and map
consistency. If it does not contain a full turn or a tracking-loss case, generate deterministic
image sequences that exercise those transitions. After a recovered turn, previously observed
structure should remain within about 20-30 cm when the sequence provides enough visual evidence;
tracking loss must stop integration and must not corrupt the map.



### Stage 4: outputs for the later navigation work

Expose simple queries, so the next stage does not need to read the grid itself:

- `nearest_obstacle(direction_deg, width_deg)` returning a distance or None

- `clearance_sectors()` returning distances for, say, left / front / right corridors about 0.8 m wide

- `cell_state(x, z)` returning free / obstacle / unknown

- `blind_zone()` and `tracking_ok` flags, so the caller can say "look down" or "turn slowly"



## 4. Pitfalls checklist (each of these will bite)



1. **Latency from buffered frames.** `cv2.VideoCapture` queues frames; a slow loop then works on old ones. Keep only the newest frame

   (a capture thread that overwrites a single slot, or `CAP_PROP_BUFFERSIZE=1`), and time-stamp frames at capture, not at processing. For offline recordings, preserve source timestamps and never substitute processing time.

2. **Python loops.** Anything per-pixel must be vectorised numpy, cv2, or numba. Profile with the existing `Benchmark` before optimising anything.

3. **Floor plane lock-on.** Facing a wall with no floor visible, a plane fit will happily lock onto the wall. Constrain the normal as described in 3.1 and

   mark the plane stale instead of accepting it.

4. **Head bob.** Pitch changes every step, so re-fit the floor every frame. A fixed camera tilt will smear the floor into obstacles and obstacles into the floor.

5. **Unknown is not free.** Weak or no depth, outside the field of view, behind the person, below the frame and beyond 3.5 m are unknown. Only floor

   actually observed (and carved space inside a solid-depth ray) is free.

6. **Range-dependent noise.** The obstacle height threshold must rise with distance (section 3.1.4). A 2 cm bump at 3 m is noise.

7. **Camera exposure.** Auto-exposure in bright windows blows out half the image. Detect and report saturation or weak-depth regions from the recording. If the camera exposes controls, use repository-supported defaults; otherwise degrade to unknown rather than requiring manual tuning.

8. **Stereo is not reliable on glass, mirrors, plain white or black surfaces.** Expect holes there. The honest output is "unknown".

9. **Coordinate frames.** Three of them (camera, gravity-aligned camera, world grid). Name variables with the frame (`p_cam`, `p_grav`, `p_world`) and unit test the transforms before building on them.

10. **Left/right swapped or wrong calibration.** `DepthMapper` has an order check. If the map looks mirrored or empty, suspect this before suspecting the new code.

11. **A viewer that costs more than the mapper.** Draw the top-down map with plain numpy and `cv2.imshow` (a 120x120 image scaled up is almost free). Only decimate and draw 3D points when a local diagnostic or test requires them.

12. **Do not overfit one clip.** Run every discovered compatible recording and every generated fixture. Thresholds must be shared; report per-recording coverage and limitations.


## 5. Things not to do

- No SLAM library, no deep-learning depth model, no GPU assumptions.

- No dense global point-cloud accumulation (memory and time grow without bound, and drift ruins it).

- No "free" without evidence. No integration while tracking is lost.

- Do not change the depth stage (confidence logic, flying-pixel removal) to fix a mapping problem unless you have shown the depth is the cause.



## 6. Further reading (optional implementation context)

- Fankhauser et al., "Probabilistic Terrain Mapping for Mobile Robots with Uncertain Localization" (elevation maps with per-cell variance)

- OctoMap (Hornung et al.), for log-odds occupancy updates

- Nav2 costmap layers (obstacle and voxel layers), for how obstacle and free evidence are marked and cleared

- libviso2 and ORB-SLAM3 stereo, as reference for stereo visual odometry

- OpenCV behavior for `calcOpticalFlowPyrLK`, `solvePnPRansac`, and `goodFeaturesToTrack`

Do not pause implementation to ask the human to look up or verify any of these references. Use the
existing dependencies and local documentation, and report uncertainty in the automated diagnostics.