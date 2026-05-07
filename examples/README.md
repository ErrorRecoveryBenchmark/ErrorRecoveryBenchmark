# Example Data

This directory contains small sample data for testing the RecoverBench pipeline without downloading the full dataset (~30 GB).

## Directory Structure

### seed_demos/

6 HDF5 files (one per task, ~6 KB each) containing clean demonstration seeds used by MimicGen. These are minimal reference demos; the full prepared datasets are in the release data package.

- coffee.hdf5
- pick_place.hdf5
- stack.hdf5
- stack_three.hdf5
- threading.hdf5
- three_piece_assembly.hdf5

### error_scenes_sample/

5 error scene pairs (NPZ + JSON) from the pick_place task, showing different error subtypes. These demonstrate the output format of the v5 error injection pipeline.

### recovery_demo_sample/

Sample recovery demonstration NPZ files from the stack task. These demonstrate the output format of the human teleoperation collection phase.

## Getting Full Data

Download the complete dataset from [Zenodo / HuggingFace Datasets] (link TBD) and set:

    export BENCHMARK_DATA=/path/to/extracted/release_data

The full dataset includes:
- seed_demos/ (6 HDF5 files)
- mimicgen_prepared/ (6 prepared HDF5 datasets, ~22 GB)
- error_scenes/ (11,004 scene pairs across 6 tasks)
- recovery_demos_human/ (973 human recovery demonstrations)
- recovery_demos_augmented/ (8,957 augmented recovery demonstrations)
