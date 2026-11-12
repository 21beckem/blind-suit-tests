import importlib.util
import subprocess
import sys

# Map 'import_name': 'pip-package-name'
REQUIRED_PACKAGES = {
    'numpy': 'numpy',
    'cv2': 'opencv-python',
    'PIL': 'pillow',
    'torch': 'torch',
    'transformers': 'transformers',
    'serial': 'pyserial',
    'matplotlib': 'matplotlib',
    'open3d': 'open3d',
    'scipy': 'scipy',
    'skimage': 'scikit-image'
}

missing = [
    pip_name
    for import_name, pip_name in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(import_name) is None
]

if missing:
    print(f'Installing missing packages: {', '.join(missing)}')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
else:
    print('All required packages are already installed.')