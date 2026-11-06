# Binocular navigation HUD

## Open on a phone without internet

Run `python vr/navigate-binocular/cams-to-phone.py` on the computer. It prints one or more local addresses such as `http://192.168.1.42:8000/`; open the address on the same network as your phone. The phone and computer need to be on the same local Wi-Fi network or hotspot, but the network does not need internet access. The script serves the page and `vr.js` on port 8000 and streams the HUD on port 5000. If it lists multiple addresses, use the Wi-Fi adapter's IPv4 address from `ipconfig`. If Windows asks for firewall access, allow it on the local/private network. You can run `python vr/navigate-binocular/phone_server.py` to test page loading without starting the camera; its HUD will correctly say depth is unavailable.

`cams-to-phone.py` turns stereo depth into a two-level 5×4 walking cue. `V_ON` and `V_OFF` in `config.py` are the single source for the two brightness values. The `1` and `0` patterns below describe which cells are on and off; for example, setting `V_ON = 0.3` dims every on cell without changing the pattern. Restart the server and reload the phone page after changing these constants. The pipeline is:

1. `stereo_obstacles.py` projects depth into a local metric `LocalMap` and estimates a floor. Obstacles and measured floor are separate; missing depth remains unknown.
2. `navigation.py` fuses sensor maps and checks five person-wide forward corridors. A near hazard in the current corridor means stop. Otherwise it prefers forward and then a known, unobstructed side corridor.
3. `grid.py` encodes the decision as four rows of `V_ON`/`V_OFF`. `cams-to-phone.py` sends the same two-dimensional JSON array used by the existing phone page.

| Row | Meaning |
| --- | --- |
| 1 | One lit column = suggested direction, left to right. `11111` = stop; `10101` = cannot judge, stop and check. |
| 2 | Blocked within the near reaction zone. |
| 3 | Blocked farther ahead. |
| 4 | Insufficient coverage to judge that direction. |

The five columns are candidate walking corridors, not five pixel-sized gaps. `LocalMap` coordinates use metres, x to the right and z forward. A new obstacle source can produce the same map; add its result to `fuse_maps([...])`. Hazard evidence takes priority over observed floor, and stale evidence is ignored. Map resolution is independent of display resolution.

## Live desktop debugging

The `Binocular Navigation Debug` window displays four synchronized views of each frame sent to the phone:

- **Camera:** green pixels matched the fitted floor; red pixels were classified as hazards.
- **Metric depth:** warm colors are near; black has no usable depth. The overlay reports the valid fraction and floor-fit status, support, estimated camera height, and floor tilt.
- **Local map:** green cells contain measured floor, red cells contain hazards, and gray cells remain unknown. White lines are the five candidate walking directions; yellow is the selected direction.
- **Exact HUD:** the 5×4 bits sent to the phone, decision reason, corridor coverage percentages, and timing. A message gap above 750 ms is highlighted because the phone may show its own unknown pattern while waiting.

When the top row is `10101`, check the floor-fit status and near/far coverage to see why no route was verified. When it is `11111`, the reason distinguishes a hazard within the stop distance from all corridors being blocked. The phone can also show `10101` by itself on a connection or frame timeout; its text status says `Depth unavailable` in that case.

## Live tuning

The `Navigation Controls` window has sliders that apply to the next processed frame:

| Slider | Effect |
| --- | --- |
| Stop distance | A center-corridor hazard closer than this triggers `11111` immediately. Other blocked-route conditions can also trigger `11111`. |
| Near distance | End of the near hazard band (HUD row 2). |
| Ahead distance | End of the ahead hazard band (HUD row 3), up to the 4 m sensor limit. |
| Path width | Width of each candidate walking corridor. |
| Min coverage | Fraction of each near/ahead corridor that must have floor or hazard evidence before it is treated as known. |
| Obstacle points per cell | Minimum depth pixels classified as hazards in a local-map cell before that cell is blocked. |
| Direction switch frames | Frames before changing between two viable directions. Immediate STOP is never delayed. |

The distance sliders enforce `stop < near < ahead` with at least 20 cm between adjacent thresholds. The effective values appear on the depth panel. Press `s` to save the current settings in `navigation_settings.json` for the next run, or `d` to restore defaults. Press `r` to capture a depth frame with the current settings, and `q` or Esc to quit. Settings saved with a capture are used when that frame is replayed.

The first thresholds are in `config.py`. They require tuning with actual captures and supervised walking trials. Floor fitting needs visible, measurable ground. A flat-looking image or zero stereo depth does not establish free space. The camera does not cover the ground directly underfoot, behind objects, or outside its field of view. The HUD must therefore be treated as a prototype walking aid, not a guarantee that any route is safe.

## Capture and replay

With the desktop preview open, press `r` to save one metric depth frame and its camera geometry in `captures/`. Press `q` or Esc to quit. Reprocess a sample without the camera:

```powershell
python vr/navigate-binocular/replay_navigation.py vr/navigate-binocular/captures/depth-....npz --show
```

The replay shows depth, the local map (green = measured floor, red = hazard, black = unknown), and the exact 5×4 HUD. Without `--show`, it prints the four binary rows. Run synthetic checks with:

```powershell
python -m unittest discover -s vr/navigate-binocular/tests -v
```

The phone page uses a discrete rendering path so blur, adaptation, noise, and display resampling do not change the code. It displays the unknown pattern if frames stop arriving or the socket disconnects. Its left/right orientation can still be calibrated with the VR display's flip and rotation settings.
