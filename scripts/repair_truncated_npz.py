#!/usr/bin/env python3
"""Recover CRC-verified complete NPY members; never reconstruct truncated data."""
import argparse
import hashlib
import io
import json
import struct
import zipfile
import zlib
from pathlib import Path

import numpy as np


def recover(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError("Destination already exists; preserve the original")
    data = source.read_bytes()
    position, recovered, omitted = 0, {}, []
    while position + 30 <= len(data) and data[position:position + 4] == b"PK\x03\x04":
        _, flags, method, _, _, crc, compressed, size, name_len, extra_len = struct.unpack_from(
            "<5H3I2H", data, position + 4)
        if flags & 9 or method not in (0, 8):
            raise ValueError("Encrypted/streaming ZIP or unsupported compression")
        name_end = position + 30 + name_len
        name = data[position + 30:name_end].decode("utf-8")
        if not name.endswith(".npy") or "/" in name or "\\" in name or name in recovered:
            raise ValueError("Unexpected or duplicate NPY member")
        extra = data[name_end:name_end + extra_len]
        if size == 0xffffffff or compressed == 0xffffffff:
            offset, found = 0, False
            while offset + 4 <= len(extra):
                tag, length = struct.unpack_from("<HH", extra, offset)
                payload = extra[offset + 4:offset + 4 + length]
                if tag == 1:
                    values = iter(struct.unpack("<" + "Q" * (len(payload) // 8), payload))
                    if size == 0xffffffff:
                        size = next(values)
                    if compressed == 0xffffffff:
                        compressed = next(values)
                    found = True
                    break
                offset += 4 + length
            if not found:
                raise ValueError("Missing ZIP64 sizes")
        start = name_end + extra_len
        end = start + compressed
        if end > len(data):
            omitted.append(name)
            break
        raw = zlib.decompress(data[start:end], -15) if method == 8 else data[start:end]
        if len(raw) != size or (zlib.crc32(raw) & 0xffffffff) != crc:
            raise ValueError(f"CRC or size mismatch: {name}")
        np.load(io.BytesIO(raw), allow_pickle=False)
        recovered[name] = raw
        position = end
    if not {"actions.npy", "states.npy"}.issubset(recovered):
        raise ValueError("Complete actions and states are required")
    actions = np.load(io.BytesIO(recovered["actions.npy"]), allow_pickle=False)
    states = np.load(io.BytesIO(recovered["states.npy"]), allow_pickle=False)
    if (actions.ndim != 2 or states.ndim != 2 or not len(actions)
            or len(states) not in (len(actions), len(actions) + 1)
            or not np.isfinite(actions).all() or not np.isfinite(states).all()):
        raise ValueError("Invalid action/state arrays")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for name, raw in recovered.items():
            out.writestr(name, raw)
    return {
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "repaired_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "preserved_npy_sha256": {n: hashlib.sha256(raw).hexdigest() for n, raw in recovered.items()},
        "omitted_truncated_members": omitted,
        "trailing_members": "Unknown: the original central directory is absent; none reconstructed",
        "num_steps": len(actions),
        "note": "Complete NPY payloads preserved byte-for-byte with CRC validation; no images synthesized; action replay not certified.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    report = recover(args.source, args.destination)
    args.destination.with_suffix(".repair.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
