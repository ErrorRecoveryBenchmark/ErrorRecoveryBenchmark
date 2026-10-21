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

The original two truncated human NPZ files have complete, CRC-verified action and
state members, but incomplete camera images. `scripts/repair_truncated_npz.py`
can create an explicit repaired copy containing only complete NPY members; it
refuses to overwrite a file or repair an incomplete action/state sequence and
records all preserved payload hashes. Release v1.1.0 uses this count-preserving repair. Do not substitute the earlier two-file-exclusion
candidate when maintaining the paper's counts. Missing images must be disclosed.


## State-rendered image supplement

The two repaired human records have separately stored, explicitly aligned image
companions in `RecoverBench_image_supplement.zip` on the dataset page. All 880
saved states are rendered without integrating actions. The surviving original
image prefixes are preserved separately. The new images have a consistent pinned
simulator view, but are not original pixels and must not be spliced into the old
image streams. They do not add trajectories or alter paper counts/results.
Use `camera_state_indices` and `state_times`, not the old four-step sampling rule.
Verify the ZIP against the corresponding entry in `SHA256SUMS`.

To regenerate a companion from a readable recovered NPZ:

```bash
python -m scripts.render_recovery_state_images recovery.npz recovery.images.npz \
  --seed-dataset /path/to/threading_d0_prepared.hdf5 \
  --task-config error_benchmark/configs/tasks/threading.yaml
```

The submitting author confirmed all-coauthor agreement with the paper's
no-competing-interests statement on October 21, 2026.
