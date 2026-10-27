# Third-Party Notices

We release the RecoverBench code, error scenes and recovery trajectories under the MIT License (see [`LICENSE`](LICENSE)). Some components keep their upstream terms: the vendored MimicGen code, one utility module adapted from it (`pose_transforms.py`) and two configuration files adapted from MimicGen templates remain under the NVIDIA Source Code License, and the clean source demonstrations from which the error scenes are generated come from the MimicGen datasets, which are released under CC BY 4.0 and are credited here and in the paper.

This file lists which files are not covered by the MIT License, reproduces the license texts that apply to them, and credits the upstream assets that RecoverBench builds on.

## 1. Summary

| Component | Where | License | Section |
|---|---|---|---|
| MimicGen code | `shared/mimicgen_workspace/mimicgen/` (git submodule of [NVlabs/mimicgen](https://github.com/NVlabs/mimicgen)) | NVIDIA Source Code License | 2 |
| `pose_transforms.py` | `error_benchmark/framework/utils/pose_transforms.py` | NVIDIA Source Code License (adapted from MimicGen `pose_utils.py`) | 2.2 |
| MimicGen experiment-config adaptations | `error_benchmark/configs/mimicgen_pick_place_1000.json`, `error_benchmark/configs/mimicgen_threading_d1_1000.json` | NVIDIA Source Code License (adapted from MimicGen config templates) | 2.3 |
| MimicGen datasets | clean source demonstrations behind the error scenes (not redistributed) | CC BY 4.0 | 3 |
| robosuite 1.4.1 | `shared/mimicgen_workspace/robosuite/` (git submodule) | MIT | 4 |
| robosuite-task-zoo | `shared/mimicgen_workspace/robosuite-task-zoo/` (git submodule) | MIT | 4 |
| robomimic | installed from upstream, not vendored | MIT (see Section 4) | 4 |
| MuJoCo 2.3.2 | installed from PyPI, not vendored | Apache 2.0 | 4 |

All other files in this repository, and the data files in the Hugging Face release (error scenes in `eval_data.tar` and `training_scenes.tar`; human, MimicGen-generated and augmented recovery trajectories in `training_data.tar`, `recovery_mimicgen_generated.tar` and `recovery_training_success_from_error.tar`), are released under the MIT License.

## 2. NVIDIA Source Code License components

### 2.1 Vendored MimicGen code

`shared/mimicgen_workspace/mimicgen/` is a git submodule of MimicGen (https://github.com/NVlabs/mimicgen); the version used for the reported results is commit `72bd767c` (v1.0.1); the applied patches are listed in `patches/`. It keeps its own `LICENSE` and file headers. MimicGen's code is released under the NVIDIA Source Code License, reproduced in full in Section 2.5. MimicGen is credited by citation in the paper:

> Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, Dieter Fox. MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations. 7th Annual Conference on Robot Learning (CoRL), 2023. https://arxiv.org/abs/2310.17596

### 2.2 `error_benchmark/framework/utils/pose_transforms.py`

- **Origin:** adapted from `mimicgen/utils/pose_utils.py` in MimicGen (vendored at `shared/mimicgen_workspace/mimicgen/mimicgen/utils/pose_utils.py`).
- **Upstream notice (retained):** "Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved. Licensed under the NVIDIA Source Code License [see LICENSE for details]."
- **License of this file:** NVIDIA Source Code License (Section 2.5). The file is not covered by the MIT License of this repository.
- **Changes:** the module provides the same ten pose functions as `pose_utils.py` (`make_pose`, `unmake_pose`, `pose_inv`, `pose_in_A_to_pose_in_B`, `quat2axisangle`, `axisangle2quat`, `quat_slerp`, `interpolate_rotations`, `interpolate_poses`, `transform_source_data_segment_using_object_pose`). It has been modified so that it imports without robosuite or MimicGen: robosuite's quaternion helpers `mat2quat` and `quat2mat` are inlined as `_mat2quat` and `_quat2mat` (the inlined helpers follow robosuite's `transform_utils`, MIT, see Section 4).

### 2.3 MimicGen experiment-config adaptations

`error_benchmark/configs/mimicgen_pick_place_1000.json` and `error_benchmark/configs/mimicgen_threading_d1_1000.json` are edited copies of MimicGen's experiment-config templates (`mimicgen/exps/templates/robosuite/pick_place.json` and `threading.json`). Most values are unchanged; we changed the experiment name, source and output paths, generation counts and seed, and the rendering and camera settings (and, for threading, the task name and one selection strategy). They remain under the NVIDIA Source Code License (Section 2.5).

### 2.4 Practical consequences

The NVIDIA Source Code License limits use of the Work and of derivative works to non-commercial use, meaning research or evaluation purposes only (its Section 3.3). The task environments named in the scene files (for example `PickPlace_D0`) are defined by the MimicGen package, which `error_benchmark/scripts/utils/script_utils.py` imports to register them. Running the benchmark simulator therefore uses MimicGen under its own license; the MIT License of RecoverBench does not change that.

### 2.5 Full text: NVIDIA Source Code License

This is the text of `shared/mimicgen_workspace/mimicgen/LICENSE`, reproduced verbatim as Section 3.1(b) of the license requires.

```
NVIDIA License

1. Definitions

“Licensor” means any person or entity that distributes its Work.
“Work” means (a) the original work of authorship made available under this license, which may include software, documentation, or other files, and (b) any additions to or derivative works  thereof  that are made available under this license.
The terms “reproduce,” “reproduction,” “derivative works,” and “distribution” have the meaning as provided under U.S. copyright law; provided, however, that for the purposes of this license, derivative works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work.
Works are “made available” under this license by including in or with the Work either (a) a copyright notice referencing the applicability of this license to the Work, or (b) a copy of this license.

2. License Grant

2.1 Copyright Grant. Subject to the terms and conditions of this license, each Licensor grants to you a perpetual, worldwide, non-exclusive, royalty-free, copyright license to use, reproduce, prepare derivative works of, publicly display, publicly perform, sublicense and distribute its Work and any resulting derivative works in any form.

3. Limitations

3.1 Redistribution. You may reproduce or distribute the Work only if (a) you do so under this license, (b) you include a complete copy of this license with your distribution, and (c) you retain without modification any copyright, patent, trademark, or attribution notices that are present in the Work.

3.2 Derivative Works. You may specify that additional or different terms apply to the use, reproduction, and distribution of your derivative works of the Work (“Your Terms”) only if (a) Your Terms provide that the use limitation in Section 3.3 applies to your derivative works, and (b) you identify the specific derivative works that are subject to Your Terms. Notwithstanding Your Terms, this license (including the redistribution requirements in Section 3.1) will continue to apply to the Work itself.

3.3 Use Limitation. The Work and any derivative works thereof only may be used or intended for use non-commercially. Notwithstanding the foregoing, NVIDIA Corporation and its affiliates may use the Work and any derivative works commercially. As used herein, “non-commercially” means for research or evaluation purposes only.

3.4 Patent Claims. If you bring or threaten to bring a patent claim against any Licensor (including any claim, cross-claim or counterclaim in a lawsuit) to enforce any patents that you allege are infringed by any Work, then your rights under this license from such Licensor (including the grant in Section 2.1) will terminate immediately.

3.5 Trademarks. This license does not grant any rights to use any Licensor’s or its affiliates’ names, logos, or trademarks, except as necessary to reproduce the notices described in this license.

3.6 Termination. If you violate any term of this license, then your rights under this license (including the grant in Section 2.1) will terminate immediately.

4. Disclaimer of Warranty.

THE WORK IS PROVIDED “AS IS” WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF 
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT. YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER THIS LICENSE. 

5. Limitation of Liability.

EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION, LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
```

## 3. MimicGen datasets (CC BY 4.0)

The clean source demonstrations from which RecoverBench generates its error scenes come from the MimicGen datasets, which are released under the Creative Commons Attribution 4.0 International license (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/).

- **Creators:** Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, Dieter Fox.
- **Source:** https://huggingface.co/datasets/amandlek/mimicgen_datasets
- **Reference:** the MimicGen paper cited in Section 2.1.
- **Files used:** the MimicGen human source demonstrations for `pick_place`, and the MimicGen `core` D0 demonstrations for the other five tasks (`stack`, `coffee`, `threading`, `stack_three`, `three_piece_assembly`). The `dataset` field of each scene JSON records the source file and the demonstration used.
- **Changes:** RecoverBench does not redistribute the MimicGen HDF5 files; `scripts/download_data.sh` can fetch the source demonstrations from the upstream repository. The released error scenes store simulator states taken from frames of these demonstrations after an injected error and a settling period, and the recovery trajectories start from such states. Both are therefore adapted material, and this notice indicates that changes were made.

The error scenes and recovery trajectories are released under the MIT License (Section 1); the attribution above is given because they are adapted from CC BY 4.0 material. CC BY 4.0 does not apply to the MimicGen code.

## 4. Other vendored and required components

### robosuite (MIT)

`shared/mimicgen_workspace/robosuite/` is a git submodule of robosuite (https://github.com/ARISE-Initiative/robosuite), pinned to version 1.4.1 at commit `c848ca84` (see `ROBOSUITE_VERSION_LOCK.txt`). The local changes used for the experiments are kept as a patch in `patches/robosuite-c848ca84-recoverbench.patch`; the patched code stays under robosuite's license. Text of `shared/mimicgen_workspace/robosuite/LICENSE`:

```
MIT License

Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This software includes the partial implementation of Deepmind Mujoco https://github.com/deepmind/mujoco. 
Deepmind Mujoco is licensed under the Apache License, Version 2.0 (the "License");
you may not use the files except in compliance with the License. 

You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
```

### robosuite-task-zoo (MIT)

`shared/mimicgen_workspace/robosuite-task-zoo/` is a git submodule of robosuite-task-zoo (https://github.com/ARISE-Initiative/robosuite-task-zoo); the version used for the reported results is commit `74eab7f8`. Text of `shared/mimicgen_workspace/robosuite-task-zoo/LICENSE`:

```
MIT License

Copyright (c) 2021 ARISE Initiative

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### robomimic and MuJoCo (not vendored)

robomimic (https://github.com/ARISE-Initiative/robomimic, version 0.3.0 in the reported runs) is used for the BC-RNN baselines and MuJoCo 2.3.2 is the physics engine. Both are installed from upstream and are not redistributed here. The paper's appendix lists robomimic as MIT and MuJoCo as Apache 2.0; consult the upstream repositories for the authoritative license texts. The appendix of the paper also lists the licenses of the other training and evaluation dependencies (openpi and the Pi 0.5 base weights, OpenVLA, NumPy, PyTorch, JAX), which are likewise not redistributed.

## 5. Questions

Please report licensing questions or missing attributions through the GitHub issue tracker: https://github.com/ErrorRecoveryBenchmark/ErrorRecoveryBenchmark/issues
