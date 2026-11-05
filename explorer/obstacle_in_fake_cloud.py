import open3d as o3d
import numpy as np
from point_cloud_generator import generate_scan

import sys
from pathlib import Path
script_dir = Path(__file__).resolve().parent
target_dir = (script_dir / '..' / 'binocular' / 'with-phone').resolve()
if str(target_dir) not in sys.path:
    sys.path.insert(0, str(target_dir))
import obstacle_scene as obs


cfg = obs.default_config()
scan = generate_scan({
    'seed': 1,
    'objects': { 'count': 8 },
    'scan': { 'omniscient_scan': True }
})
result = obs.analyze_point_cloud(o3d, scan.points, cfg)
groups = obs.scene_groups(o3d, scan.points, scan.colors, result, cfg)
center = np.median(scan.points, axis=0)

print('\nViewer: X = points, G = planes, V = obstacles | ctrl+drag = pan')
obs.show_scene(o3d, groups, cfg['point_size'], center)



