"""Release regressions: private content exclusion and exact dependency pins."""
import subprocess
from pathlib import Path
import pytest
from scripts.prepare_github_release import prepare_release, public_file
from scripts.download_data import parse_checksums, safe_extract, download, sha256


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def test_export_preserves_pins_and_excludes_internal_untracked_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init")
    (source / ".gitignore").write_text("error_benchmark/internal_test.py\n")
    for name in ["README.md", ".gitmodules", "docs/CAMERA_READY_TODO.md", "ideas/graph.yaml",
                 "deliverables/private.txt", "error_benchmark/core.py",
                 "THIRD_PARTY_NOTICES.md", "CITATION.cff", "patches/robosuite.patch"]:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("public" if name in {"README.md", ".gitmodules", "error_benchmark/core.py"} else "internal")
    git(source, "add", ".")
    sha = "c848ca848020d0c4ccdd10c5056bd06f2a195ba2"
    submodule = "shared/mimicgen_workspace/robosuite"
    git(source, "update-index", "--add", "--cacheinfo", f"160000,{sha},{submodule}")
    git(source, "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-m", "fixture")
    (source / "error_benchmark/internal_test.py").write_text("from workflow import internal_tool")
    out = tmp_path / "release"
    result = prepare_release(source, out)
    assert result["submodules"][submodule] == sha
    assert sha in git(out, "ls-files", "--stage", submodule)
    assert (out / "README.md").is_file()
    for name in ["THIRD_PARTY_NOTICES.md", "CITATION.cff", "patches/robosuite.patch"]:
        assert (out / name).read_bytes() == (source / name).read_bytes()
    assert not (out / "docs/CAMERA_READY_TODO.md").exists()
    assert not (out / "ideas").exists()
    assert not (out / "deliverables").exists()
    assert not (out / "error_benchmark/internal_test.py").exists()
    assert git(out, "rev-list", "--all", "--count") == "0"
    with pytest.raises(ValueError, match="already exists"):
        prepare_release(source, out)


def test_internal_paths_are_not_public():
    for path in [".companion/x.py", "error_benchmark/outputs/x.json", "docs/CAMERA_READY_TODO.md"]:
        assert not public_file(Path(path))


def test_checksums_require_every_selected_archive():
    with pytest.raises(ValueError, match="Missing checksums"):
        parse_checksums("a" * 64 + "  eval_data.tar\n")
    with pytest.raises(ValueError, match="Duplicate"):
        parse_checksums(("a" * 64 + "  eval_data.tar\n") * 2, ["eval_data.tar"])


def test_npz_recovery_preserves_complete_payloads_and_omits_truncated_images(tmp_path):
    import io
    import zipfile
    import numpy as np
    from scripts.repair_truncated_npz import recover
    original = tmp_path / "original.npz"
    with zipfile.ZipFile(original, "w", compression=zipfile.ZIP_STORED) as z:
        for name, value in [("actions", np.ones((3, 7))), ("states", np.ones((4, 45))),
                            ("camera_images", np.ones((4, 8, 8, 3)))]:
            stream = io.BytesIO()
            np.save(stream, value)
            z.writestr(name + ".npy", stream.getvalue())
    with zipfile.ZipFile(original) as z:
        image_start = z.getinfo("camera_images.npy").header_offset
        expected = {n: z.read(n) for n in ["actions.npy", "states.npy"]}
    truncated = tmp_path / "broken.npz"
    truncated.write_bytes(original.read_bytes()[:image_start + 90])
    repaired = tmp_path / "repaired.npz"
    report = recover(truncated, repaired)
    assert report["omitted_truncated_members"] == ["camera_images.npy"]
    with zipfile.ZipFile(repaired) as z:
        assert {n: z.read(n) for n in z.namelist()} == expected
    bad = bytearray(truncated.read_bytes())
    bad[14] ^= 1  # Corrupt the first member's stored CRC.
    truncated.write_bytes(bad)
    with pytest.raises(ValueError, match="CRC"):
        recover(truncated, tmp_path / "must_not_exist.npz")
    assert not (tmp_path / "must_not_exist.npz").exists()


def test_archive_rejects_traversal_before_writing_anything(tmp_path):
    import io
    import tarfile
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as tar:
        for name in ["good.txt", "../outside.txt"]:
            entry = tarfile.TarInfo(name)
            entry.size = 4
            tar.addfile(entry, io.BytesIO(b"test"))
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(archive, tmp_path / "extracted")
    assert not (tmp_path / "extracted/good.txt").exists()
    assert not (tmp_path / "outside.txt").exists()


def test_verified_download_from_local_mirror(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"a verified download")
    target = tmp_path / "target.bin"
    download(source.as_uri(), target, sha256(source))
    assert target.read_bytes() == source.read_bytes()
    # An existing correct file needs no network access.
    download("https://invalid.invalid", target, sha256(source))
    # A server that ignores Range must restart rather than append duplicate bytes.
    second = tmp_path / "second.bin"
    second.with_name("second.bin.part").write_bytes(b"partial")
    download(source.as_uri(), second, sha256(source))
    assert second.read_bytes() == source.read_bytes()


def test_data_root_resolves_downloaded_seed_layout(tmp_path, monkeypatch):
    pytest.importorskip("robosuite")
    from error_benchmark.scripts.utils.script_utils import load_task_registry
    monkeypatch.setenv("BENCHMARK_DATA", str(tmp_path))
    assert Path(load_task_registry("stack")["dataset_path"]) == tmp_path / "seed_demos/stack.hdf5"


def test_inventory_reports_unreadable_demo_instead_of_silently_dropping_it(tmp_path):
    import io
    import tarfile
    from scripts.build_release_metadata import inspect_archive
    archive = tmp_path / "training_data.tar"
    with tarfile.open(archive, "w") as tar:
        entry = tarfile.TarInfo("recovery_demos_human/stack/position_error_D0/broken.npz")
        entry.size = 6
        tar.addfile(entry, io.BytesIO(b"broken"))
    report, _, _ = inspect_archive(archive, tmp_path / "index.csv", "demo")
    assert report["records"] == 1
    assert len(report["invalid_records"]) == 1
    assert "broken.npz" in (tmp_path / "index.csv").read_text()
