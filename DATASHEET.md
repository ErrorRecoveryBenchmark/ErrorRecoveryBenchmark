# Datasheet for RecoverBench

A datasheet for datasets ([Gebru et al., 2021](https://arxiv.org/abs/1803.09010)) for the RecoverBench dataset and benchmark.

> RecoverBench measures recovery from injected post-failure simulator states, alongside clean-start task success. Published results depend on the exact policy checkpoint, data version, evaluation scenes, and success criterion; see `EVALUATION.md`.

---

## 1. Motivation

**For what purpose was the dataset created?**
RecoverBench was created to enable systematic evaluation of robot manipulation policies under post-failure execution states — a regime existing benchmarks (LIBERO, RLBench, CALVIN, robosuite, MimicGen, DROID, BEDLAM) do not measure. Existing benchmarks score policies from clean initial states only, producing capability estimates that do not generalize to deployment, where execution errors (slips, collisions, misgrasps) are routine. Recent ad-hoc recovery work (SC-VLA, CycleVLA, FLARE, RoboFAC, FailSafe) has begun to study this regime, but each uses its own 2–4 error types, success criteria, and simulator setups, making cross-paper comparison impossible. RecoverBench provides the missing standardization layer: a 12-skill / 24-subtype error taxonomy, deterministically reproducible error scenes, a from-error-state evaluation protocol paired with the standard from-clean protocol, and a recovery training corpus.

**Who created the dataset and on behalf of which entity?**
Created by Zijia Tang, Ganlong Zhao, Xingping Chen, Junye Chen, Zihao Mo, Zedi Wang, Guanbin Li. See `CITATION.cff` for the affiliations in the author-provided camera-ready source. Project: https://github.com/ErrorRecoveryBenchmark/ErrorRecoveryBenchmark.

**Who funded the creation of the dataset?**
The author-provided camera-ready acknowledgement credits the National Natural Science Foundation of China (62322608), Shenzhen Loop Area Institute (FPF10120260001), and computational resources from National Supercomputer Center in Guangzhou. The authors declare no competing interests; all-coauthor agreement was confirmed by the submitting author on October 21, 2026.

---

## 2. Composition

**What do the instances represent?**
Three core instance types. The camera-ready corpus additionally includes 668 generated trajectories in `recovery_mimicgen_generated.tar`, giving 10,598 historical entries. `recovery_training_success_from_error.tar` contains the separate 739 trajectories used by the reported recovery-training runs and is not added to that corpus count. Counts below describe archive entries, not a claim that every file passes replay. Release v1.1.1 stores image observations alongside numerical trajectory arrays. Image alignment and rendering provenance are documented in DATA_FORMAT.md and record metadata. The corpus contains 10,598 trajectories; the 739 HDF5 training trajectories remain a separate artifact.


- **Error scenes**: a `(post_sim_state, RNG state, environment fingerprint)` snapshot representing a deterministically reproducible failure state (each scene = one NPZ + one JSON sidecar).
- **Human recovery demonstrations** (973 archive entries): `(states, actions)` recordings of human recovery attempts; stored success flags do not certify successful replay.
- **MimicGen-augmented recovery demonstrations** (8,957 total): synthetically generated recovery trajectories produced by warping human demonstrations onto new error scenes via 2,532 scene-warping trajectories and 6,425 recorded-action replay trajectories, as detailed in the camera-ready appendix.

**How many instances are there in total, and how is the dataset structured?**

The core consists of three archives (see the [dataset card on HuggingFace](https://huggingface.co/datasets/ErrorRecovery/ErrorRecoveryBenchmark) and `SHA256SUMS`):

| Archive | Contents | Purpose |
|---|---|---|
| `eval_data.tar` | 1,360 error scenes (136 task × subtype combinations × 10; 2,720 files) | **Evaluation set** (frozen) |
| `training_scenes.tar` | 5,038 error scenes | **Training scene pool** — disjoint from the evaluation set (verified: zero shared scene IDs, zero identical post-error state fingerprints); recovery demos were collected on scenes from this pool |
| `training_data.tar` | `recovery_demos_human/` 973 NPZ + `recovery_demos_augmented/` 8,957 NPZ | Recovery-demonstration training data |

Clean-trajectory seed demos are downloaded from MimicGen's public HuggingFace
repository by `scripts/download_data.sh` (not redistributed here).

Per-task scene counts — evaluation set: coffee 200 · pick_place 240 · stack 240 ·
stack_three 240 · threading 200 · three_piece_assembly 240; training pool:
coffee 200 · pick_place 2,080 · stack 2,079 · stack_three 240 · threading 199 ·
three_piece_assembly 240.

Error subtypes: 12 Error Skills × 2 difficulty degrees (D0: mild displacement < 10–15 cm; D1: severe displacement ≥ 10–15 cm with rotation) = 24 subtypes, organized into 5 Recovery Behavior Groups (RBGs A/B/C/D/E). See [`README.md`](README.md) for the taxonomy table.

**Modalities per instance:**
- RGB: 224×224 agentview camera images (and other camera views available via robosuite config)
- Proprioception: 7-DOF joint states + gripper state (continuous, 0–1)
- Object poses: 3D positions + quaternions in `(w, x, y, z)` order
- Depth: optional, available via `camera_depths` parameter
- Language: per-RBG instruction templates (e.g., RBG_E: "correct the position error and resume the task")

**Does the dataset contain all possible instances or is it a sample?**
Sample. The error-skill × task × difficulty grid covers 24 subtypes × 6 tasks = 144 cells. The frozen evaluation release covers 136 cells (1,360 scenes). This observed coverage is distinct from the number of theoretically applicable cells; per-task counts are listed above.

**What data does each instance consist of?**
Error scenes: NPZ with `sim_state` (pre-injection) and `post_sim_state` (post-injection stable state); JSON with metadata (error skill, degree, injection frame, source dataset reference, RNG seed, environment fingerprint). Recovery demos: NPZ with `states`, `actions` (6 OSC pose components + 1 gripper command), `recovery_subtypes` (RBG-segmented subtask labels).

**Are there labels?**
Yes. Error scenes are labeled with error skill name, degree, RBG, task, and source clean trajectory. Recovery demos are labeled with `recovery_subtypes` per timestep, identifying the active recovery sub-skill (e.g., `re_grasp`, `re_orient`).

**Are there recommended data splits?**
Yes — the split is physical: `eval_data.tar` is the frozen 1,360-scene evaluation set; `training_scenes.tar` (scene pool) and `training_data.tar` (`recovery_demos_human/`, `recovery_demos_augmented/`) are the training side. Train/eval disjointness is verified at the level of scene IDs and SHA-256 fingerprints of the canonical post-error simulator state.

**Are there errors, sources of noise, or redundancies?**
- Augmentation success rates vary widely by subtype (~9.6% mean on stack, with some subtypes at 0%); failed augmentations are filtered out before inclusion. Successful augmentations may still contain near-duplicate trajectories from the same human demo source.
- Human teleoperation demonstrations contain operator variability. Recorded success and replay flags describe the collection pipeline; the current archive audit is structural and does not certify successful replay of every released demonstration.

**Does the dataset rely on external resources?**
Yes. The data was generated using vendored forks of:
- `robosuite` (commit c848ca84, MuJoCo 2.3.2)
- `mimicgen`
- `robosuite-task-zoo`

These are bundled in `shared/mimicgen_workspace/` so the dataset can be regenerated/replayed without external dependency drift. See `ROBOSUITE_VERSION_LOCK.txt` for version pinning.

**Does the dataset contain sensitive, personal, or confidential information?**
No. All data is simulated (MuJoCo); there are no human subjects, no personal data, no images of identifiable individuals. The human teleoperator's identity is not recorded in any instance.

---

## 3. Collection Process

**How was the data acquired?**
Three pipelines:

1. **Error scene generation (v5 Error-Skill + Context-Replay pipeline):** clean trajectories from `seed_demos/` are loaded; for each (task, error skill, degree) cell, the pipeline replays the clean trajectory up to a target injection frame, applies the error skill's `inject()` method (parameterized by degree), simulates to a stable state, and stores `(sim_state, post_sim_state, RNG, fingerprint)`. Fully automated and deterministic.

2. **Human recovery demonstration collection:** operators use a SpaceMouse with Sawyer (`pick_place`) or Panda (other tasks) after an injected error. Collection and replay-validation code are provided, but historical files must not be assumed to pass all checks in the current code. The paper distinguishes stored simulator-state restoration from action-replay reproducibility.

3. **MimicGen scene-configuration augmentation:** human demonstrations are RBG-segmented, then re-targeted onto new error scenes within the same subtype via MimicGen's scene-configuration warping. Augmentation feasibility is verified on a sample of N=10 candidate scenes per subtype before bulk generation; only successful augmentations are retained.

**Who collected the data?**
The RecoverBench authors. The human teleoperator(s) are members of the author team.

**Over what time frame was the data collected?**
2026-Q1 through 2026-Q2 (data version 1.0.0, dated 2026-05).

**Were any ethical review processes conducted?**
The dataset is **simulation-only** (MuJoCo 2.3.2 with Sawyer and Panda arms in robosuite). It contains no human-subjects data, no personally identifiable information, no real-world imagery, no third-party copyrighted content, and no health, demographic, or behavioral data about identifiable individuals. The teleoperator(s) are authors of the project and consented to release of the recorded trajectories; no operator identity is recorded in any released instance. Under common research-ethics frameworks (e.g., U.S. Common Rule §46.102) this work does not constitute human-subjects research, and IRB / ethics-board review was therefore not required. We have nonetheless audited all released trajectories to confirm no incidental personal data is embedded in metadata or filenames.

---

## 4. Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling done?**
The released augmentation corpus uses two-subtask segmentation at gripper close (plus a 15-frame buffer); RBG-aware segmentation is an optional mode. Augmentations were checked for success at generation time. The current audit checks archive readability, action/state dimensions and finite values; it does not certify successful action replay of every record. The shipped augmented stack example fails replay under the pinned stack, as recorded in `examples/recovery_demo_sample/replay_validation.json`.

**Was the raw data saved?**
Published archive revisions are preserved in dataset history. All complete numerical payloads are retained byte-for-byte. Image sources, state alignment and validation are documented in record metadata; no numerical states were synthesized.

---

## 5. Uses

**What tasks has the dataset been used for?**
Within the project: evaluating Pi0.5 (task-specific fine-tuned) and BC-RNN baselines on the from-error-state vs. from-clean-state gap; training error-aware VLA fine-tunes on the human + augmented recovery corpus.

**What are the recommended uses?**
- Recovery evaluation under the RecoverBench gap-reporting protocol (paired Clean Success Rate and Recovery Success Rate; see [`EVALUATION.md`](EVALUATION.md))
- Recovery skill learning and imitation
- RBG-aware imitation (e.g., conditioning on RBG label)
- Error-aware VLA fine-tuning (foundation models trained on clean data, fine-tuned on RecoverBench's recovery corpus)

**What uses are inappropriate / discouraged?**
- Treating `post_sim_state` as a clean initial state in benchmarks that do not disclose this provenance — this would defeat the comparability the dataset is built for.
- Reporting from-error numbers without paired from-clean numbers — this would re-create the mis-measurement problem RecoverBench exists to fix.
- Drawing real-world deployment conclusions from v1.0.0 alone: the data is simulated only.

**Are there tasks for which the dataset should not be used?**
The dataset is not suitable for training general-purpose policies on its own (it is recovery-specific) and should not be used as a substitute for clean-state training data.

---

## 6. Distribution

**Will the dataset be distributed to third parties?**
Yes. Public open release alongside the NeurIPS 2026 Evaluations and Datasets camera-ready release.

**How will it be distributed?**
HuggingFace Hub ([ErrorRecovery/ErrorRecoveryBenchmark](https://huggingface.co/datasets/ErrorRecovery/ErrorRecoveryBenchmark)). The code release lives in this repository.

**Under what license?**
MIT License. Vendored dependencies in `shared/mimicgen_workspace/` retain their original licenses (robosuite, MimicGen, and robosuite-task-zoo have their own license files; MimicGen uses the NVIDIA Source Code License).

**Are there any restrictions or terms of use?**
RecoverBench-original code and data use MIT; upstream and adapted components retain the terms in `THIRD_PARTY_NOTICES.md`. Researchers are *requested* (not required) to follow the gap-reporting protocol so leaderboard numbers remain comparable across papers.

---

## 7. Maintenance

**Who is supporting / hosting / maintaining the dataset?**
The author team listed in `CITATION.cff` maintains the dataset. Use https://github.com/ErrorRecoveryBenchmark/ErrorRecoveryBenchmark/issues for issues and errata. Corresponding contact: Guanbin Li, liguanbin@mail.sysu.edu.cn; first-author contact: Zijia Tang, zt81@duke.edu.

**How will errata or corrections be communicated?**
Via Hugging Face revision history and the GitHub issue tracker. Archive checksums identify exact versions. The camera-ready correction is being prepared; see docs/RELEASE.md and metadata/release_inventory.json.

**Will the dataset be updated?**
Yes. Planned future versions:
- v1.1: additional baseline evaluations (more VLAs and imitation methods)
- v2: real-robot extension (additional embodiments beyond simulated Sawyer and Panda)
- v2: additional task families and error skills based on community feedback

**How can others contribute to the dataset?**
By submitting new evaluations to the leaderboard (see [`EVALUATION.md`](EVALUATION.md) for the submission template), reporting issues via the project repository, or proposing new error skills / RBGs through the project's issue tracker.
