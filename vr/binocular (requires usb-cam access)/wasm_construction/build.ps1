param([switch]$SkipConfigure)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$opencv = Join-Path $root 'vendor/opencv'
$emsdk = Join-Path $root 'vendor/emsdk'
$build = Join-Path $root 'build'
$dist = Join-Path $root 'dist'
$vsTools = Get-ChildItem 'C:\Program Files\Microsoft Visual Studio' -Directory |
    ForEach-Object { Get-ChildItem $_.FullName -Directory } |
    ForEach-Object { Join-Path $_.FullName 'Common7/IDE/CommonExtensions/Microsoft/CMake' } |
    Where-Object { Test-Path (Join-Path $_ 'CMake/bin/cmake.exe') } |
    Select-Object -First 1
if (!$vsTools) { throw 'Visual Studio CMake installation not found.' }
$cmakeDir = Join-Path $vsTools 'CMake/bin'
$ninjaDir = Join-Path $vsTools 'Ninja'

if (!(Test-Path (Join-Path $opencv 'platforms/js/build_js.py'))) {
    throw 'OpenCV 4.12.0 source missing. Clone it to vendor/opencv.'
}
if (!(Test-Path (Join-Path $emsdk 'emsdk_env.ps1'))) {
    throw 'Emscripten 4.0.20 missing. Install it in vendor/emsdk.'
}
if (!(Test-Path (Join-Path $cmakeDir 'cmake.exe')) -or !(Test-Path (Join-Path $ninjaDir 'ninja.exe'))) {
    throw 'CMake and Ninja were not found in Visual Studio.'
}

# OpenCV 4.12 hard-codes C++11 for its JS target, but Emscripten 4.0.20
# Embind requires C++17. Apply the local source fix on every clean checkout.
$jsCmakeFile = Join-Path $opencv 'modules/js/CMakeLists.txt'
$jsCmakeText = Get-Content -LiteralPath $jsCmakeFile -Raw
if ($jsCmakeText.Contains('add_definitions("-std=c++11")')) {
    $jsCmakeText = $jsCmakeText.Replace('add_definitions("-std=c++11")', 'add_definitions("-std=c++17")')
}
$jsCmakeText = $jsCmakeText.Replace(' -s DEMANGLE_SUPPORT=1', '')
$jsCmakeText | Set-Content -LiteralPath $jsCmakeFile -NoNewline

$env:EMSDK_QUIET = '1'
. (Join-Path $emsdk 'emsdk_env.ps1') | Out-Null
$env:PATH = "$cmakeDir;$ninjaDir;$env:PATH"
$toolchain = Join-Path $emsdk 'upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake'
$script = Join-Path $opencv 'platforms/js/build_js.py'
$args = @(
    $script, $build,
    '--opencv_dir', $opencv,
    '--config', (Join-Path $root 'opencv_sgbm.config.py'),
    '--disable_single_file', '--simd', '--config_only',
    "--cmake_option=-DCMAKE_TOOLCHAIN_FILE=$toolchain",
    '--cmake_option=-DCMAKE_CXX_STANDARD=17',
    '--cmake_option=-DBUILD_opencv_dnn=OFF',
    '--cmake_option=-DBUILD_opencv_photo=OFF',
    '--cmake_option=-DBUILD_opencv_video=OFF',
    '--cmake_option=-DBUILD_EXAMPLES=OFF',
    '--cmake_option=-DBUILD_TESTS=OFF',
    '--cmake_option=-DBUILD_PERF_TESTS=OFF',
    '--cmake_option=-G', '--cmake_option=Ninja'
)
if ($SkipConfigure) { $args += '--skip_config' }
& python @args
if ($LASTEXITCODE -ne 0) { throw "OpenCV configuration failed ($LASTEXITCODE)" }
& (Join-Path $ninjaDir 'ninja.exe') -C $build -j 4 opencv.js
if ($LASTEXITCODE -ne 0) { throw "OpenCV build failed ($LASTEXITCODE)" }
New-Item -ItemType Directory -Force -Path $dist | Out-Null
Copy-Item -LiteralPath (Join-Path $build 'bin/opencv.js') -Destination $dist
Copy-Item -LiteralPath (Join-Path $build 'bin/opencv_js.wasm') -Destination $dist
Copy-Item -LiteralPath (Join-Path $opencv 'LICENSE') -Destination (Join-Path $dist 'OPENCV-LICENSE.txt')
Write-Host "WASM assets are in $dist"
