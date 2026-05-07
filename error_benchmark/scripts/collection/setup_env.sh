#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Setup conda environment for recovery demo collection
# NOTE: This script is designed for the portable collection package.
#       Run from the package root, not from the repository.
#       See TUTORIAL.md for packaging instructions.
# ═══════════════════════════════════════════════════════════════
# Run this ONCE on the target machine (with display + SpaceMouse).
#
# Prerequisites:
#   - Miniconda/Anaconda installed
#   - System packages: libhidapi-dev libglfw3-dev (for SpaceMouse + rendering)
#     Install with: sudo apt-get install -y libhidapi-dev libglfw3-dev
#
# Usage:
#   bash setup_env.sh

set -euo pipefail

ENV_NAME="recovery_collect"
PYTHON_VERSION="3.10"
MUJOCO_VERSION="2.3.2"

echo "═══════════════════════════════════════════════════════"
echo " Recovery Collection Environment Setup"
echo "═══════════════════════════════════════════════════════"

# Check conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install Miniconda first."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create environment
echo "Creating conda env: $ENV_NAME (Python $PYTHON_VERSION)..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo "Installing packages..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Core dependencies
pip install numpy scipy "mujoco==$MUJOCO_VERSION" h5py pyyaml

# Rendering and SpaceMouse
pip install opencv-python glfw hidapi

# Utility packages
pip install tqdm imageio numba

# Install robosuite locally from the exact locked upstream commit.
# Keep mimicgen from the bundled local copy.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACK_ROOT="$SCRIPT_DIR"
LOCK_FILE="$PACK_ROOT/ROBOSUITE_VERSION_LOCK.txt"

ROBOSUITE_INSTALL_URL="git+https://github.com/ARISE-Initiative/robosuite.git@c848ca848020d0c4ccdd10c5056bd06f2a195ba2#egg=robosuite"
if [ -f "$LOCK_FILE" ]; then
    ROBOSUITE_INSTALL_URL=$(grep '^robosuite_install_url=' "$LOCK_FILE" | cut -d= -f2-)
fi

echo "Installing robosuite from locked upstream commit..."
pip install "$ROBOSUITE_INSTALL_URL"

if [ -d "$PACK_ROOT/shared/mimicgen_workspace/mimicgen" ]; then
    echo "Installing mimicgen from portable pack..."
    pip install -e "$PACK_ROOT/shared/mimicgen_workspace/mimicgen"
else
    echo "WARNING: bundled mimicgen not found at $PACK_ROOT/shared/mimicgen_workspace/mimicgen"
    echo "  You may need to install mimicgen manually."
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo " Setup complete!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Verify installation:"
echo "  conda activate $ENV_NAME"
echo "  python -c \"import mujoco; import robosuite; print('OK')\""
echo ""
echo "SpaceMouse Linux setup (if needed):"
echo "  1. Create /etc/udev/rules.d/99-spacemouse.rules:"
echo '     SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", MODE="0666"'
echo "  2. sudo udevadm control --reload-rules && sudo udevadm trigger"
echo "  3. Verify: python -c \"import hid; print([d for d in hid.enumerate() if d['vendor_id']==0x256f])\""
