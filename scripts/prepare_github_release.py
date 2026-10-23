#!/usr/bin/env python3
"""Export a public working tree, preserving the repository's pinned gitlinks."""
import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT_FILES = {
    "README.md", "DATASHEET.md", "EVALUATION.md", "LICENSE", "Makefile",
    "ROBOSUITE_VERSION_LOCK.txt", "pyproject.toml", "croissant.json",
    ".gitmodules", ".gitattributes", ".gitignore", "collect_all.sh", "setup_env.sh",
    "CITATION.cff", "THIRD_PARTY_NOTICES.md",
}
PUBLIC_DOCS = {"TUTORIAL.md", "HF_DATASET_CARD.md", "RELEASE.md", "DATA_FORMAT.md"}
EXCLUDED_PARTS = {
    ".git", "__pycache__", ".pytest_cache", "outputs", "logs", "archive",
    ".agents", ".claude", ".codex", ".companion", "ideas", "deliverables",
}
# Match private machine paths, not legitimate author names or attribution.
PRIVATE_PATH = re.compile(r"(?:/HOME/[A-Za-z0-9_.-]+/[^\s\"']+|/XYFS\d+/[^\s\"']+|[A-Z]:[\\/]Users[\\/][^\s\"']+)")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def public_file(relative):
    parts = relative.parts
    if any(part in EXCLUDED_PARTS for part in parts):
        return False
    if relative.name in {"AGENTS.md", "CLAUDE.md"} or relative.suffix in {".pyc", ".log"}:
        return False
    if len(parts) == 1:
        return relative.name in ROOT_FILES
    if parts[0] == "docs":
        return len(parts) == 2 and relative.name in PUBLIC_DOCS
    if parts[0] in {"error_benchmark", "scripts", "examples", "patches", "metadata"}:
        return relative.suffix in {".py", ".sh", ".yaml", ".yml", ".json", ".md", ".txt", ".npz", ".hdf5", ".tsv", ".csv", ".patch"}
    return relative.as_posix() == "shared/mimicgen_workspace/README.md"


def prepare_release(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if output == source or output in source.parents:
        raise ValueError("Output cannot contain the source repository")
    links = []
    for line in git(source, "ls-tree", "-r", "HEAD").splitlines():
        mode, kind, rest = line.split(" ", 2)
        sha, path = rest.split("\t", 1)
        if mode == "160000":
            links.append((sha, path))
    selected = []
    for directory, dirs, files in os.walk(source):
        parent = Path(directory)
        dirs[:] = [d for d in dirs if d not in EXCLUDED_PARTS
                   and (parent / d).resolve() != output
                   and not (parent / d).is_symlink()
                   and not any((parent / d) == source / p for _, p in links)
                   and (parent != source or d in {"error_benchmark", "scripts", "examples", "docs", "patches", "shared", "metadata"})]
        for name in files:
            path = parent / name
            if path.is_symlink() or not public_file(path.relative_to(source)):
                continue
            selected.append(path)
    relative_names = [path.relative_to(source).as_posix() for path in selected]
    ignored_result = subprocess.run(
        ["git", "-C", str(source), "check-ignore", "--stdin", "-z"],
        input="\0".join(relative_names) + "\0", text=True, capture_output=True,
    )
    if ignored_result.returncode not in (0, 1):
        raise ValueError(f"Cannot check release exclusions: {ignored_result.stderr}")
    ignored = set(ignored_result.stdout.split("\0"))
    selected = [path for path in selected if path.relative_to(source).as_posix() not in ignored]
    for path in selected:
        if path.suffix not in {".npz", ".hdf5"}:
            if PRIVATE_PATH.search(path.read_text(encoding="utf-8", errors="replace")):
                raise ValueError(f"Private machine path in {path.relative_to(source)}")
    output.mkdir(parents=True)
    for path in selected:
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    git(output, "init", "-b", "main")
    git(output, "config", "core.autocrlf", "false")
    git(output, "add", ".")
    for sha, path in links:
        (output / path).mkdir(parents=True, exist_ok=True)
        git(output, "update-index", "--add", "--cacheinfo", f"160000,{sha},{path}")
    return {"files": len(selected), "submodules": dict((p, sha) for sha, p in links)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        print(prepare_release(args.source, args.output))
        print("Review staged files, commit, then run git submodule update --init --recursive.")
    except (ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Release export failed: {exc}\n")
