# Vendored Dependencies

This directory contains patched forks of robotics libraries required by RecoverBench.

## Submodules

After cloning, initialize submodules:

```bash
git submodule update --init --recursive
```

| Submodule | Upstream | Pinned Commit |
|-----------|----------|---------------|
| robosuite | https://github.com/ARISE-Initiative/robosuite | c848ca848020d0c4ccdd10c5056bd06f2a195ba2 |
| mimicgen | https://github.com/NVlabs/mimicgen | (main branch at time of release) |
| robosuite-task-zoo | https://github.com/ARISE-Initiative/robosuite-task-zoo | (main branch at time of release) |

## Local Patches

After submodule init, apply the RecoverBench patch:

```bash
cd shared/mimicgen_workspace
git apply ../../patches/robosuite_recoverbench.patch
```

The patch adds:
- `robosuite/robosuite/macros_private.py` — GPU rendering config for MuJoCo EGL

## Installation

```bash
pip install -e shared/mimicgen_workspace/robosuite
pip install -e shared/mimicgen_workspace/mimicgen
pip install -e shared/mimicgen_workspace/robosuite-task-zoo
```

Or use the provided setup script:

```bash
bash setup_env.sh
```
