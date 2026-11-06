# Binocular navigation HUD

## Open on a phone without internet

Run `python vr/binocular/cams-to-phone.py` on the computer. It prints one or more local addresses such as `http://192.168.1.42:8000/`; open the address on the same network as your phone. The phone and computer need to be on the same local Wi-Fi network or hotspot, but the network does not need internet access. The script serves the page and `vr.js` on port 8000 and streams the HUD on port 5000. If it lists multiple addresses, use the Wi-Fi adapter's IPv4 address from `ipconfig`. If Windows asks for firewall access, allow it on the local/private network. You can run `python vr/binocular/phone_server.py` to test page loading without starting the camera; its HUD will correctly say depth is unavailable.

`cams-to-phone.py` turns stereo depth into a binary 5×4 walking cue. The pipeline is:

1. `stereo_obstacles.py` projects depth into a local metric `LocalMap` and estimates a floor. Obstacles and measured floor are separate; missing depth remains unknown.
2. `navigation.py` fuses sensor maps and checks five person-wide forward corridors. A near hazard in the current corridor means stop. Otherwise it prefers forward and then a known, unobstructed side corridor.
3. `grid.py` encodes the decision as four rows of `0`/`1`. `cams-to-phone.py` sends the same two-dimensional JSON array used by the existing phone page.

| Row | Meaning |
| --- | --- |
| 1 | One lit column = suggested direction, left to right. `11111` = stop; `10101` = cannot judge, stop and check. |
| 2 | Blocked within the near reaction zone. |
| 3 | Blocked farther ahead. |
| 4 | Insufficient coverage to judge that direction. |

The five columns are candidate walking corridors, not five pixel-sized gaps. `LocalMap` coordinates use metres, x to the right and z forward. A new obstacle source can produce the same map; add its result to `fuse_maps([...])`. Hazard evidence takes priority over observed floor, and stale evidence is ignored. Map resolution is independent of display resolution.

The first thresholds are in `config.py`. They require tuning with actual captures and supervised walking trials. Floor fitting needs visible, measurable ground. A flat-looking image or zero stereo depth does not establish free space. The camera does not cover the ground directly underfoot, behind objects, or outside its field of view. The HUD must therefore be treated as a prototype walking aid, not a guarantee that any route is safe.

## Capture and replay

With the desktop preview open, press `r` to save one metric depth frame and its camera geometry in `captures/`. Press `q` or Esc to quit. Reprocess a sample without the camera:

```powershell
python vr/binocular/replay_navigation.py vr/binocular/captures/depth-....npz --show
```

The replay shows depth, the local map (green = measured floor, red = hazard, black = unknown), and the exact 5×4 HUD. Without `--show`, it prints the four binary rows. Run synthetic checks with:

```powershell
python -m unittest discover -s vr/binocular/tests -v
```

The phone page uses a discrete rendering path so blur, adaptation, noise, and display resampling do not change the code. It displays the unknown pattern if frames stop arriving or the socket disconnects. Its left/right orientation can still be calibrated with the VR display's flip and rotation settings.
