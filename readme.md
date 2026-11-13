# Blind Suit Tests

This repository contains experiments and tools for binocular computer vision,
depth mapping, and navigation.

## Setup

You need Python 3 and a webcam or recorded camera data for the live examples.
From the repository root, run the setup script for your operating system:

```powershell
.\activate.bat
```

On macOS or Linux:

```bash
./activate.sh
```

The script creates `.venv`, activates it, and installs missing Python packages.

## Run the stereo depth example

From `binocular/cloud-scan`, run:

```powershell
..\..\.venv\Scripts\python.exe image_depth.py
```

This loads the configured stereo images and opens a depth preview. To use a
camera instead, run `live_depth.py`:

```powershell
..\..\.venv\Scripts\python.exe live_depth.py
```

Press **Esc** or **Q** to close the preview. Edit
[`binocular/cloud-scan/config.json`](binocular/cloud-scan/config.json) to
change camera, image, or depth settings.

For the obstacle-map examples and module details, see the
[cloud-scan README](binocular/cloud-scan/README.md).

## Run tests

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s binocular/cloud-scan/tests -v
```

The tests use synthetic images and mocked camera/window APIs, so a physical
camera is not required.