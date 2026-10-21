#!/usr/bin/env python3
"""Download, verify and safely extract the frozen RecoverBench release."""
import argparse
import hashlib
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

BASE_URL = "https://huggingface.co/datasets/ErrorRecovery/ErrorRecoveryBenchmark/resolve/main"
ARCHIVES = ("eval_data.tar", "training_scenes.tar", "training_data.tar",
            "recovery_mimicgen_generated.tar", "recovery_training_success_from_error.tar")
TASKS = ("pick_place", "stack", "stack_three", "coffee", "threading", "three_piece_assembly")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksums(text, required=ARCHIVES):
    checksums = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *]([^\r\n]+)", line)
        if not match:
            raise ValueError("Malformed SHA256SUMS line")
        digest, name = match.groups()
        if name in checksums:
            raise ValueError(f"Duplicate checksum: {name}")
        checksums[name] = digest.lower()
    missing = set(required) - checksums.keys()
    if missing:
        raise ValueError(f"Missing checksums: {', '.join(sorted(missing))}")
    return checksums


def download(url, target, expected=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and expected and sha256(target) == expected:
        return
    partial = target.with_name(target.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request, timeout=60) as response:
        resume = offset and getattr(response, "status", None) == 206
        if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError(f"Incorrect download range for {target.name}")
        with partial.open("ab" if resume else "wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
    if expected and sha256(partial) != expected:
        raise ValueError(f"Checksum mismatch for {target.name}; retained {partial.name} for inspection")
    partial.replace(target)


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            target = (destination / member.name).resolve()
            if (path.is_absolute() or ".." in path.parts or "\\" in member.name
                    or ":" in member.name or not (member.isfile() or member.isdir())
                    or (target != destination and destination not in target.parents)):
                raise ValueError(f"Unsafe archive entry: {member.name}")
        # All entries and existing destination symlinks are checked before extraction.
        tar.extractall(destination, members=members)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", type=Path, default=Path("data"))
    parser.add_argument("--base-url", default=BASE_URL, help="Versioned release URL or local file:// mirror")
    parser.add_argument("--archives", nargs="+", choices=ARCHIVES, default=list(ARCHIVES))
    parser.add_argument("--seeds", action="store_true", help="Also download upstream MimicGen seed demos for regeneration")
    parser.add_argument("--verify-only", action="store_true", help="Check existing archives without network or extraction")
    args = parser.parse_args()
    try:
        args.destination.mkdir(parents=True, exist_ok=True)
        sums = args.destination / "SHA256SUMS"
        if not args.verify_only:
            download(args.base_url.rstrip("/") + "/SHA256SUMS", sums)
        expected = parse_checksums(sums.read_text(), args.archives)
        for name in args.archives:
            path = args.destination / name
            if not args.verify_only:
                print(f"Downloading/verifying {name}", flush=True)
                download(args.base_url.rstrip("/") + "/" + name, path, expected[name])
            if sha256(path) != expected[name]:
                raise ValueError(f"Checksum mismatch: {name}")
        if not args.verify_only:
            for name in args.archives:
                safe_extract(args.destination / name, args.destination)
            if args.seeds:
                for task in TASKS:
                    download(f"https://huggingface.co/datasets/amandlek/mimicgen_datasets/resolve/main/source/{task}.hdf5",
                             args.destination / "seed_demos" / f"{task}.hdf5")
        print(f"Verified {len(args.archives)} archives in {args.destination.resolve()}")
    except (OSError, ValueError, tarfile.TarError) as exc:
        parser.exit(1, f"Dataset download failed: {exc}\n")


if __name__ == "__main__":
    main()
