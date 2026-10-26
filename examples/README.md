# Small runnable samples

This directory includes five pick_place error-scene NPZ/JSON pairs and one human
and one augmented recovery NPZ. The recovery files are copied without modification
from the core training archive; `recovery_demo_sample/provenance.json` records
archive member names, sizes, action counts, and SHA-256 hashes.

Run a CPU-only format check from the repository root:

```bash
python scripts/inspect_examples.py
```

This checks arrays, trajectory lengths, JSON metadata, and sample hashes. See
[DATA_FORMAT.md](../docs/DATA_FORMAT.md) for optional fields and array conventions.

Upstream seeds are fetched into `seed_demos/` with `python scripts/download_data.py data --seeds`. The five release archives
total approximately 9.7 GB compressed; upstream seeds and prepared training formats
require additional disk space. See [RELEASE.md](../docs/RELEASE.md).
