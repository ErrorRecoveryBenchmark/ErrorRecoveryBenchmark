#!/usr/bin/env python3
"""Check the shipped scene and trajectory formats without a simulator or GPU."""
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1] / "examples"
    scenes = list((root / "error_scenes_sample").glob("*.npz"))
    for path in scenes:
        metadata = json.loads(path.with_suffix(".json").read_text())
        with np.load(path, allow_pickle=False) as arrays:
            assert arrays["post_sim_state"].ndim == 1, path
            assert np.isfinite(arrays["post_sim_state"]).all(), path
        assert metadata["scene_id"], path
    provenance = json.loads((root / "recovery_demo_sample/provenance.json").read_text())
    for record in provenance:
        path = root / "recovery_demo_sample" / record["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], path
        with np.load(path, allow_pickle=False) as arrays:
            n = len(arrays["actions"])
            assert n == record["num_steps"] and n > 0, path
            assert len(arrays["states"]) in (n, n + 1), path
            assert np.isfinite(arrays["actions"]).all(), path
    assert scenes and provenance, "Sample data missing"
    print(f"Validated {len(scenes)} scenes and {len(provenance)} demonstrations (format only, not simulation replay).")


if __name__ == "__main__":
    main()
