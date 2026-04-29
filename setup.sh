#!/bin/bash
# Setup for M4 OpenCV resize benchmark.
# Run from the same folder as M4benchmark.py:
#   chmod +x setup.sh && ./setup.sh

set -e

echo "=== M4 benchmark setup ==="
echo

PYARCH=$(python3 -c 'import platform; print(platform.machine())')
if [ "$PYARCH" != "arm64" ]; then
    echo "❌ Your python3 is $PYARCH, not arm64."
    echo "   You're running through Rosetta. Install an arm64 Python first:"
    echo "     brew install python"
    echo "   Then re-run this script."
    exit 1
fi
echo "✅ Python is arm64"
PYVER=$(python3 --version)
echo "   $PYVER"
echo

if ! command -v brew >/dev/null 2>&1; then
    echo "⚠️  Homebrew not found. The pyvips path will be skipped."
    echo "   Install Homebrew from https://brew.sh if you want pyvips coverage."
    SKIP_VIPS=1
else
    echo "✅ Homebrew found"
fi
echo

if [ -z "$SKIP_VIPS" ]; then
    if brew list vips >/dev/null 2>&1; then
        echo "✅ libvips already installed"
    else
        echo "📦 Installing libvips via Homebrew (~30 seconds)…"
        brew install vips
    fi
fi
echo

echo "📦 Installing Python packages…"
python3 -m pip install --upgrade pip --quiet
python3 -m pip install --quiet \
    "opencv-python>=4.9" \
    "numpy>=1.26" \
    "pillow>=10" \
    "matplotlib>=3.8" \
    "torch" \
    "pyobjc-framework-Quartz" \
    "pyobjc-framework-Metal"

if [ -z "$SKIP_VIPS" ]; then
    python3 -m pip install --quiet "pyvips"
fi

echo
echo "=== Verifying installs ==="
python3 - <<'PY'
def check(name, importer):
    try:
        importer()
        print(f"  ✅ {name}")
    except Exception as e:
        print(f"  ❌ {name}: {type(e).__name__}: {e}")

check("opencv-python", lambda: __import__("cv2"))
check("numpy",         lambda: __import__("numpy"))
check("Pillow",        lambda: __import__("PIL"))
check("matplotlib",    lambda: __import__("matplotlib"))
check("PyTorch",       lambda: __import__("torch"))

import torch
if torch.backends.mps.is_available():
    print("  ✅ PyTorch MPS backend available")
else:
    print("  ⚠️  PyTorch MPS NOT available (the GPU path won't run)")

check("pyobjc Quartz", lambda: __import__("Quartz"))
check("pyobjc Metal",  lambda: __import__("Metal"))

try:
    import pyvips
    print(f"  ✅ pyvips {'.'.join(str(pyvips.version(i)) for i in range(3))}")
except Exception as e:
    print(f"  ⚠️  pyvips: {e}")

import cv2
print(f"  ℹ️  OpenCV {cv2.__version__}, OpenCL available: {cv2.ocl.haveOpenCL()}")
PY

echo
echo "=== Ready ==="
echo "Now run:    python3 M4benchmark.py"
echo
echo "Tip: for cleanest numbers, plug in to power, close Chrome/Slack/Zoom,"
echo "     and don't run on battery saver. Total runtime: ~3-5 minutes."
