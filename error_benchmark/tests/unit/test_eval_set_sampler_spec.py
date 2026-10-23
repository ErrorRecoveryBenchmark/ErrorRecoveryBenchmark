"""Spec tests for eval_set_sampler: deterministic stratified sampling
over the validation scene pool (the frozen eval set).

Authoritative protocol (docs/项目全景总结.md): the eval set is a sample of
the ~130k M12 validation scenes; per (task, subtype) draw N (min(N, available)).
"""
import json

from error_benchmark.scripts.training import eval_set_sampler as ess


def _recs(task, subtype, n, prefix="s"):
    return [
        {"task": task, "subtype": subtype, "scene_id": f"{prefix}_{i:04d}",
         "path": f"/v/{task}/{subtype}/{prefix}_{i:04d}.json"}
        for i in range(n)
    ]


def test_per_cell_capped_at_n():
    recs = _recs("stack", "grasp_misalignment_D0", 100) + _recs("coffee", "wrong_object_D1", 10, "c")
    m = ess.stratified_sample(recs, n=50, seed=42)
    cells = m["cells"]
    assert cells["stack/grasp_misalignment_D0"]["available"] == 100
    assert cells["stack/grasp_misalignment_D0"]["selected"] == 50
    # under-N cell takes all
    assert cells["coffee/wrong_object_D1"]["selected"] == 10
    assert m["total_selected"] == 60


def test_deterministic_same_seed():
    recs = _recs("stack", "grasp_misalignment_D0", 100)
    a = ess.stratified_sample(recs, n=20, seed=42)
    b = ess.stratified_sample(recs, n=20, seed=42)
    assert [s["scene_id"] for s in a["scenes"]] == [s["scene_id"] for s in b["scenes"]]


def test_different_seed_differs():
    recs = _recs("stack", "grasp_misalignment_D0", 100)
    a = ess.stratified_sample(recs, n=20, seed=1)
    b = ess.stratified_sample(recs, n=20, seed=2)
    assert [s["scene_id"] for s in a["scenes"]] != [s["scene_id"] for s in b["scenes"]]


def test_no_duplicates_and_subset():
    recs = _recs("stack", "grasp_misalignment_D0", 100)
    m = ess.stratified_sample(recs, n=30, seed=7)
    ids = [s["scene_id"] for s in m["scenes"]]
    assert len(ids) == len(set(ids))
    assert set(ids).issubset({r["scene_id"] for r in recs})


def test_under_n_cell_stable_sorted_order():
    recs = _recs("coffee", "wrong_object_D0", 5)
    m = ess.stratified_sample(recs, n=50, seed=42)
    ids = [s["scene_id"] for s in m["scenes"]]
    assert ids == sorted(ids)


def test_manifest_provenance_fields():
    recs = _recs("stack", "grasp_misalignment_D0", 100)
    m = ess.stratified_sample(recs, n=10, seed=42)
    assert m["seed"] == 42
    assert m["n_per_cell"] == 10
    assert m["num_cells"] == 1
    assert "cells" in m and "scenes" in m


def test_independent_cells_unaffected_by_others():
    """A cell's selection must not change when other cells are added/removed
    (per-cell deterministic seeding)."""
    base = _recs("stack", "grasp_misalignment_D0", 100)
    m1 = ess.stratified_sample(base, n=20, seed=42)
    m2 = ess.stratified_sample(base + _recs("coffee", "wrong_object_D1", 80, "c"), n=20, seed=42)
    pick1 = [s["scene_id"] for s in m1["scenes"] if s["task"] == "stack"]
    pick2 = [s["scene_id"] for s in m2["scenes"] if s["task"] == "stack"]
    assert pick1 == pick2


def test_subtype_from_v5_filename():
    assert ess._subtype_from_stem("v5_collision_eef_object_D0_00731d8acf70") == "collision_eef_object_D0"
    assert ess._subtype_from_stem("v5_position_error_D0_abc123") == "position_error_D0"
    assert ess._subtype_from_stem("v5_grasp_misalignment_D1_deadbeef") == "grasp_misalignment_D1"
    assert ess._subtype_from_stem("scene_0001") is None


def test_discover_v5_filename_fast_path(tmp_path):
    """v5 scenes get their subtype from the filename, with NO JSON read needed."""
    d = tmp_path / "stack" / "scenes"
    d.mkdir(parents=True)
    sid = "v5_grasp_misalignment_D1_deadbeef"
    (d / f"{sid}.json").write_text("{}")   # empty content: subtype must come from filename
    (d / f"{sid}.npz").write_bytes(b"")
    recs = ess.discover_records(str(tmp_path))
    assert len(recs) == 1
    assert recs[0]["subtype"] == "grasp_misalignment_D1"
    assert recs[0]["task"] == "stack"


def test_discover_and_write(tmp_path):
    task, sub = "stack", "grasp_misalignment_D0"
    d = tmp_path / task / "scenes"
    d.mkdir(parents=True)
    for i in range(3):
        sid = f"scene_{i:04d}"
        (d / f"{sid}.json").write_text(json.dumps(
            {"labels": {"error_name": "grasp_misalignment", "degree": "D0", "subtype_id": sub}}))
        (d / f"{sid}.npz").write_bytes(b"")  # paired npz is required
    # a json without a paired npz must be ignored
    (d / "orphan.json").write_text(json.dumps({"labels": {"subtype_id": sub}}))

    recs = ess.discover_records(str(tmp_path))
    assert len(recs) == 3
    assert all(r["task"] == task and r["subtype"] == sub for r in recs)

    out = tmp_path / "eval_set_manifest.json"
    m = ess.stratified_sample(recs, n=2, seed=42)
    ess.write_manifest(m, str(out))
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["total_selected"] == 2
