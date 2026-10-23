# Camera-ready release preparation

The camera-ready bundle is being reconciled against the frozen archives. Treat
`metadata/release_inventory.json` as the observed archive inventory, and check its
`invalid_records` and train/eval overlap fields before publication. A successful
Croissant schema check alone does not certify the underlying demonstrations.

## Download and inspect

```bash
python scripts/download_data.py data
python scripts/download_data.py data --verify-only
python scripts/inspect_examples.py
```

The planned five camera-ready archives total approximately 9.7 GB compressed. `--archives eval_data.tar`
selects just the evaluation scenes. `--seeds` additionally downloads upstream
MimicGen seed demonstrations into `data/seed_demos/`. Configure seed dataset paths
in `error_benchmark/configs/task_registry.yaml` for generation. Download requires
all selected archives to have unique entries in `SHA256SUMS`; all hashes are checked
before extraction. Downloads use `.part` files and resume when the server supports it.

## Rebuild metadata from real files

```bash
python scripts/build_release_metadata.py --eval /path/to/eval_data.tar \
  --training-scenes /path/to/training_scenes.tar \
  --demos /path/to/training_data.tar --output metadata
```

This preserves the source archives, emits CSV indexes, checks simulator-state and
scene-ID overlap, and fails if a demonstration cannot be read. Do not silently
remove bad records or change published counts. Resolve them from an intact source,
or document an author-approved version change.

## Export public code

```bash
python scripts/prepare_github_release.py /path/to/new-release-directory
```

The exporter includes the current working files in public code/data/documentation
paths, excludes internal notes and execution outputs, and checks for private
machine paths. It initializes an independent Git index with the exact three
submodule commits from the source repository. Review the staged diff, commit it,
then run `git submodule update --init --recursive` and the tests. It never pushes.
The supplementary PDF is not exported automatically: attach the author-approved
camera-ready supplement after checking its references and anonymization.

## Final release checks

Confirm the accepted author list, affiliations, funding and disclosures. Keep
paper statistics distinct from archive counts and later experimental additions.
Publish the validated archives, indexes, checksums, dataset card, license, and
Croissant together; verify access without credentials. Upload the same Croissant
to OpenReview. Freeze a code commit and dataset revision for reproducibility.

## Author materials reconciled on October 21

The supplied latest paper already distinguishes 973 human, 668 generated, and
8,957 augmented trajectories (10,598 total), and separately describes the
739-trajectory HDF5 artifact used by the reported Recovery policies. Do not add
those 739 to the corpus total. Authors and affiliations are in `CITATION.cff`.
The current downloader supports all five archives.

## Integrated image observations

Version 1.1.1 distributes all release data in five archives. The two camera arrays
rendered from saved states are stored inside their human demonstration NPZs in
`training_data.tar`. Their `camera_state_indices`, `camera_state_times`,
`camera_source` and `camera_metadata` fields specify alignment and provenance.
See `docs/DATA_FORMAT.md`. Complete numerical payloads and corpus counts remain
unchanged. Previous revisions remain accessible in dataset history.

To regenerate images from saved states:

```bash
python -m scripts.render_recovery_state_images recovery.npz images.npz \
  --seed-dataset /path/to/threading_d0_prepared.hdf5 \
  --task-config error_benchmark/configs/tasks/threading.yaml
python -m scripts.integrate_recovery_images recovery.npz images.npz images.json integrated.npz
```

The integration command verifies source hashes, demo identity, image dimensions,
state indices and timestamps, preserves existing NPY payloads byte-for-byte, and
refuses to overwrite an output or replace existing image fields.

The submitting author confirmed all-coauthor agreement with the paper's
no-competing-interests statement on October 21, 2026.
