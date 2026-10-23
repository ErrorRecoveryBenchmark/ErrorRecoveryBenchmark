# Small runnable samples

This directory includes five pick_place error-scene NPZ/JSON pairs and one human
and one augmented recovery NPZ. The recovery files are copied without modification
from the core training archive; `recovery_demo_sample/provenance.json` records
archive member names, sizes, action counts, and SHA-256 hashes.

Run a CPU-only format check from the repository root:

```bash
python scripts/inspect_examples.py
```

This checks arrays, trajectory lengths, JSON metadata, and sample hashes. It does
not run a policy or prove successful simulator replay. See
[DATA_FORMAT.md](../docs/DATA_FORMAT.md) for optional fields and array conventions.

`seed_demos/` is an empty placeholder; upstream seeds are not bundled. To obtain
them, run `python scripts/download_data.py data --seeds`. The three core archives
are approximately 9 GB compressed; upstream seeds and prepared training formats
require additional disk space. See [RELEASE.md](../docs/RELEASE.md).

## Replay validation scope

A local check with MuJoCo 2.3.2 / robosuite 1.4.1 restored both samples' states.
The human coffee sample replayed with a final end-effector deviation below 1 mm.
The augmented stack sample showed approximately 12 cm of final end-effector drift,
above the existing 5 cm replay threshold. It remains as a transparent historical
format example, not a certified replay-success example. This does not establish
full-dataset replay success. See `recovery_demo_sample/replay_validation.json`.
