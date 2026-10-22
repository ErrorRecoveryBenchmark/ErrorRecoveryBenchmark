#!/usr/bin/env python3
"""Add verified, state-aligned images to a new copy of a recovery NPZ."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


def integrate(source, images, report_path, destination):
    source, images, report_path, destination = map(Path, (source, images, report_path, destination))
    if destination.exists():
        raise ValueError("Destination exists; refusing to overwrite")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for path, key in ((source, "source_npz_sha256"), (images, "image_npz_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != report[key]:
            raise ValueError(f"Source hash mismatch: {path.name}")
    with np.load(source, allow_pickle=False) as demo, np.load(images, allow_pickle=False) as camera:
        if "camera_images" in demo:
            raise ValueError("Source already has images")
        if str(demo["demo_id"]) != str(camera["demo_id"]) or str(demo["demo_id"]) != report["demo_id"]:
            raise ValueError("Demo identity mismatch")
        indices = camera["camera_state_indices"]
        if not np.array_equal(indices, np.arange(len(demo["states"]))) or indices.dtype.kind not in "iu":
            raise ValueError("Images must cover every saved state exactly once")
        if not np.array_equal(camera["state_times"], demo["states"][:, 0]):
            raise ValueError("Image/state timestamp mismatch")
        frames = camera["camera_images"]
        if frames.dtype != np.uint8 or frames.shape != (len(indices), report["resolution"], report["resolution"], 3):
            raise ValueError("Invalid image shape or dtype")
        metadata = dict(report)
        metadata["alignment"] = "camera_images[i] corresponds to states[camera_state_indices[i]]; camera_state_times contains saved simulator times"
    def npy(value):
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        return stream.getvalue()
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(images) as img:
        if src.testzip() or img.testzip():
            raise ValueError("Input ZIP CRC mismatch")
        original = {name: src.read(name) for name in src.namelist()}
        additions = {
            "camera_images.npy": img.read("camera_images.npy"),
            "camera_state_indices.npy": img.read("camera_state_indices.npy"),
            "camera_state_times.npy": img.read("state_times.npy"),
            "camera_source.npy": npy(np.array("state_rerender")),
            "camera_metadata.npy": npy(np.array(json.dumps(metadata, sort_keys=True))),
        }
        if original.keys() & additions.keys():
            raise ValueError("Refusing to replace existing fields")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as out:
            for name, payload in (original | additions).items():
                out.writestr(name, payload)
    with zipfile.ZipFile(destination) as out:
        assert out.testzip() is None
        assert all(out.read(name) == payload for name, payload in original.items())
    return {"sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "frames": len(indices), "preserved_members": len(original),
            "camera_source": "state_rerender"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "images", "report", "destination"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(integrate(args.source, args.images, args.report, args.destination), indent=2))
