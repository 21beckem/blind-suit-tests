#!/usr/bin/env bash

set -euo pipefail

# Always operate from the directory containing this script.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if [[ ! -f .venv/bin/activate ]]; then
    if [[ -e .venv ]]; then
        echo ".venv is incomplete. Removing it before recreating the environment..."
        rm -rf .venv
    fi
    echo ".venv not found. Creating virtual environment..."
    if ! python3 -m venv .venv >/dev/null 2>&1; then
        python_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        rm -rf .venv
        cat >&2 <<EOF

Python's venv support is not installed on this system.
Install it with:

    sudo apt install python${python_version}-venv -y

Then run this script again.
EOF
        exit 1
    fi
else
    echo ".venv already exists. Skipping creation."
fi

echo "Activating virtual environment..."
# Activation only affects this script's shell, which is sufficient for the
# package installation below. To keep it active in your current shell, run:
#   source ./setup-venv.sh
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Virtual environment setup complete."
