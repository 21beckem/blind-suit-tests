"""Small OpenCV.js API for the browser stereo depth test (OpenCV 4.12.0)."""

core = {"": [], "Algorithm": []}
imgproc = {
    "": ["cvtColor", "remap", "resize", "createCLAHE", "applyColorMap"],
    "CLAHE": ["apply"],
}
calib3d = {
    "": ["initUndistortRectifyMap"],
    "StereoMatcher": ["compute"],
    "StereoSGBM": ["create"],
}
features2d = {
    "Feature2D": ["detectAndCompute"],
    "ORB": ["create"],
    "DescriptorMatcher": ["match"],
    "BFMatcher": ["create"],
}

white_list = makeWhiteList([core, imgproc, calib3d, features2d])

