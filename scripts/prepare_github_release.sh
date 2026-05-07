#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Prepare a clean GitHub release of RecoverBench
#
# This script creates a clean copy suitable for GitHub upload,
# replacing vendored source trees with git submodules.
#
# Usage:
#   bash scripts/prepare_github_release.sh /path/to/output_dir
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

OUTPUT_DIR="${1:-./github_release}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RELEASE_ROOT="$(dirname "$SCRIPT_DIR")"

echo "═══════════════════════════════════════════════════════"
echo " RecoverBench GitHub Release Preparation"
echo " Source: $RELEASE_ROOT"
echo " Output: $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════"

if [ -d "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory already exists: $OUTPUT_DIR"
    echo "  Remove it first or choose a different path."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ─── Step 1: Copy code (excluding vendored repos and archive) ───
echo "[1/5] Copying source code..."
rsync -a --exclude='shared/mimicgen_workspace/robosuite/' \
         --exclude='shared/mimicgen_workspace/mimicgen/' \
         --exclude='shared/mimicgen_workspace/robosuite-task-zoo/' \
         --exclude='shared/mimicgen_workspace/archive/' \
         --exclude='shared/mimicgen_workspace/.claude/' \
         --exclude='__pycache__/' \
         --exclude='*.pyc' \
         --exclude='.pytest_cache/' \
         --exclude='error_benchmark/outputs/' \
         --exclude='examples/seed_demos/*.hdf5' \
         --exclude='examples/error_scenes_sample/*.npz' \
         --exclude='examples/error_scenes_sample/*.json' \
         --exclude='examples/recovery_demo_sample/*.npz' \
         "$RELEASE_ROOT/" "$OUTPUT_DIR/"

# ─── Step 2: Initialize git repo with submodules ───
echo "[2/5] Initializing git repository..."
cd "$OUTPUT_DIR"
git init
git add .gitmodules

# Add submodules at pinned commits
ROBOSUITE_COMMIT="c848ca848020d0c4ccdd10c5056bd06f2a195ba2"

echo "[3/5] Adding submodules..."
mkdir -p shared/mimicgen_workspace
git submodule add https://github.com/ARISE-Initiative/robosuite.git shared/mimicgen_workspace/robosuite
cd shared/mimicgen_workspace/robosuite && git checkout "$ROBOSUITE_COMMIT" && cd "$OUTPUT_DIR"

git submodule add https://github.com/NVlabs/mimicgen.git shared/mimicgen_workspace/mimicgen
git submodule add https://github.com/ARISE-Initiative/robosuite-task-zoo.git shared/mimicgen_workspace/robosuite-task-zoo

# ─── Step 3: Create patches directory ───
echo "[4/5] Creating patches..."
mkdir -p patches
cat > patches/robosuite_macros_private.py << 'PATCH_EOF'
MUJOCO_GPU_RENDERING = False
PATCH_EOF

# ─── Step 4: Summary ───
echo "[5/5] Done!"
echo ""
echo "═══════════════════════════════════════════════════════"
echo " GitHub release prepared at: $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  cd $OUTPUT_DIR"
echo "  git add -A"
echo "  git commit -m 'Initial release of RecoverBench v1.0'"
echo "  git remote add origin git@github.com:YOUR_ORG/RecoverBench.git"
echo "  git push -u origin main"
echo ""
echo "File count: $(find . -type f | grep -v '.git/' | wc -l)"
echo "Total size: $(du -sh . --exclude='.git' | cut -f1)"
