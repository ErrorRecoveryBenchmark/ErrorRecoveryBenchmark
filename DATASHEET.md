# Datasheet for RecoverBench

A datasheet for datasets ([Gebru et al., 2021](https://arxiv.org/abs/1803.09010)) for the RecoverBench dataset and benchmark.

> Robot manipulation benchmarks evaluate policies exclusively from clean initial states. In deployment, errors are not the exception — they are the steady state after the first slip, collision, or misgrasp. Existing benchmarks therefore systematically overestimate manipulation capability: they measure performance on a thin slice of the state distribution policies actually face. RecoverBench corrects this by re-grounding evaluation in *post-failure states*. Across 6 manipulation tasks, the best task-specific fine-tuned VLA we evaluate (Pi0.5) achieves a competitive clean-state success rate but recovers from only **3.7%** of injected error scenarios — and **0%** on `coffee`. The gap, not the headline number, is the contribution.

---

## 1. Motivation

**For what purpose was the dataset created?**
RecoverBench was created to enable systematic evaluation of robot manipulation policies under post-failure execution states — a regime existing benchmarks (LIBERO, RLBench, CALVIN, robosuite, MimicGen, DROID, BEDLAM) do not measure. Existing benchmarks score policies from clean initial states only, producing capability estimates that do not generalize to deployment, where execution errors (slips, collisions, misgrasps) are routine. Recent ad-hoc recovery work (SC-VLA, CycleVLA, FLARE, RoboFAC, FailSafe) has begun to study this regime, but each uses its own 2–4 error types, success criteria, and simulator setups, making cross-paper comparison impossible. RecoverBench provides the missing standardization layer: a 12-skill / 24-subtype error taxonomy, deterministically reproducible error scenes, a from-error-state evaluation protocol paired with the standard from-clean protocol, and a recovery training corpus.

**Who created the dataset and on behalf of which entity?**
The dataset was created by the RecoverBench authors for academic research and open release. Author identities and institutional affiliations are anonymized for the NeurIPS 2026 D&B double-blind review period and will be disclosed in the camera-ready version.

**Who funded the creation of the dataset?**
Funding sources and grant numbers are anonymized for the NeurIPS 2026 D&B double-blind review period. Both will be disclosed in the camera-ready version of this datasheet.

---

## 2. Composition

**What do the instances represent?**
Three instance types:
- **Error scenes** (11,004 total): a `(sim_state, post_sim_state, RNG seed, environment fingerprint)` tuple representing a deterministically reproducible failure state.
- **Human recovery demonstrations** (973 total): full `(states, actions, recovery_subtypes)` trajectories of a human teleoperator recovering from an error scene to task completion.
- **MimicGen-augmented recovery demonstrations** (8,957 total): synthetically generated recovery trajectories produced by warping human demonstrations onto new error scenes via MimicGen scene-configuration warping.

**How many instances are there in total, and how is the dataset structured?**

Top-level partitioning (see [`release_data/README.md`](../release_data/README.md) for full inventory):

| Partition | Count | Purpose |
|---|---|---|
| `error_scenes/` | 11,004 NPZ + JSON pairs | Evaluation pool |
| `recovery_demos_human/` | 973 NPZ files | Training pool (human) |
| `recovery_demos_augmented/` | 8,957 NPZ files | Training pool (augmented) |
| `mimicgen_prepared/` | 7 HDF5 files | Source data for MimicGen pipeline |
| `seed_demos/` | 6 HDF5 files | Clean-trajectory seeds |

Per-task error-scene counts: coffee 200 · pick_place 2,310 · stack 2,310 · stack_three 240 · threading 199 · three_piece_assembly 240.

Error subtypes: 12 Error Skills × 2 difficulty degrees (D0: mild displacement < 10–15 cm; D1: severe displacement ≥ 10–15 cm with rotation) = 24 subtypes, organized into 5 Recovery Behavior Groups (RBGs A/B/C/D/E). See `release_code/README.md` for the taxonomy table.

**Modalities per instance:**
- RGB: 224×224 agentview camera images (and other camera views available via robosuite config)
- Proprioception: 7-DOF joint states + gripper state (continuous, 0–1)
- Object poses: 3D positions + quaternions in `(w, x, y, z)` order
- Depth: optional, available via `camera_depths` parameter
- Language: per-RBG instruction templates (e.g., RBG_E: "correct the position error and resume the task")

**Does the dataset contain all possible instances or is it a sample?**
Sample. The error-skill × task × difficulty grid covers 24 subtypes × 6 tasks = 144 cells, of which 138 are valid (some skill–task combinations are physically inapplicable). Each valid cell is sampled multiple times (per-task counts above).

**What data does each instance consist of?**
Error scenes: NPZ with `sim_state` (pre-injection) and `post_sim_state` (post-injection stable state); JSON with metadata (error skill, degree, injection frame, source dataset reference, RNG seed, environment fingerprint). Recovery demos: NPZ with `states`, `actions` (7-DOF: 6 joint + 1 gripper), `recovery_subtypes` (RBG-segmented subtask labels).

**Are there labels?**
Yes. Error scenes are labeled with error skill name, degree, RBG, task, and source clean trajectory. Recovery demos are labeled with `recovery_subtypes` per timestep, identifying the active recovery sub-skill (e.g., `re_grasp`, `re_orient`).

**Are there recommended data splits?**
Yes — see "Standard Splits" in [`release_data/README.md`](../release_data/README.md): the full `error_scenes/` directory is the evaluation pool; `recovery_demos_human/` and `recovery_demos_augmented/` are training pools (used either independently or jointly, with disclosure).

**Are there errors, sources of noise, or redundancies?**
- Augmentation success rates vary widely by subtype (~9.6% mean on stack, with some subtypes at 0%); failed augmentations are filtered out before inclusion. Successful augmentations may still contain near-duplicate trajectories from the same human demo source.
- Human teleoperation demonstrations contain operator variability; they are validated by action-replay (reload error state, replay recorded actions, confirm task success) before inclusion.

**Does the dataset rely on external resources?**
Yes. The data was generated using vendored forks of:
- `robosuite` (commit c848ca84, MuJoCo 2.3.2)
- `mimicgen`
- `robosuite-task-zoo`

These are bundled in `release_code/shared/mimicgen_workspace/` so the dataset can be regenerated/replayed without external dependency drift. See `release_code/ROBOSUITE_VERSION_LOCK.txt` for version pinning.

**Does the dataset contain sensitive, personal, or confidential information?**
No. All data is simulated (MuJoCo); there are no human subjects, no personal data, no images of identifiable individuals. The human teleoperator's identity is not recorded in any instance.

---

## 3. Collection Process

**How was the data acquired?**
Three pipelines:

1. **Error scene generation (v5 Error-Skill + Context-Replay pipeline):** clean trajectories from `seed_demos/` are loaded; for each (task, error skill, degree) cell, the pipeline replays the clean trajectory up to a target injection frame, applies the error skill's `inject()` method (parameterized by degree), simulates to a stable state, and stores `(sim_state, post_sim_state, RNG, fingerprint)`. Fully automated and deterministic.

2. **Human recovery demonstration collection:** an operator teleoperates a Sawyer arm in robosuite via SpaceMouse from each error scene's `post_sim_state`, attempting to recover and complete the original task. Demonstrations are recorded only if (a) the environment's `_check_success()` returns true for 10 consecutive frames, AND (b) action-replay validation passes (reloading the initial error state and replaying recorded actions reproduces task success).

3. **MimicGen scene-configuration augmentation:** human demonstrations are RBG-segmented, then re-targeted onto new error scenes within the same subtype via MimicGen's scene-configuration warping. Augmentation feasibility is verified on a sample of N=10 candidate scenes per subtype before bulk generation; only successful augmentations are retained.

**Who collected the data?**
The RecoverBench authors. The human teleoperator(s) are members of the author team.

**Over what time frame was the data collected?**
2026-Q1 through 2026-Q2 (data version 1.0.0, dated 2026-05).

**Were any ethical review processes conducted?**
The dataset is **simulation-only** (MuJoCo 2.3.2 with the Sawyer arm in robosuite). It contains no human-subjects data, no personally identifiable information, no real-world imagery, no third-party copyrighted content, and no health, demographic, or behavioral data about identifiable individuals. The teleoperator(s) are authors of the project and consented to release of the recorded trajectories; no operator identity is recorded in any released instance. Under common research-ethics frameworks (e.g., U.S. Common Rule §46.102) this work does not constitute human-subjects research, and IRB / ethics-board review was therefore not required. We have nonetheless audited all released trajectories to confirm no incidental personal data is embedded in metadata or filenames.

---

## 4. Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling done?**
Yes:
- **Success-hold validation:** every recorded human recovery demo must show `_check_success() == true` for 10 consecutive frames; episodes that flash success and fail are discarded.
- **Action-replay validation:** every recorded demo is replayed from the initial error state; demos whose replay does not reproduce task success are discarded.
- **RBG segmentation:** each demo is segmented into MimicGen-compatible subtask sequences using the RBG's recovery primitive ordering (e.g., RBG_A: `retract → re_orient → re_grasp → re_lift → re_transport → re_place`).
- **Augmentation preflight:** before bulk MimicGen augmentation of a subtype, feasibility is tested on N=10 target error scenes; subtypes that fail preflight are not augmented.

**Was the raw data saved?**
Raw teleoperation streams are retained internally but not part of the public release; the public release contains only validated, post-segmentation NPZ files.

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
Yes. Public open release alongside the NeurIPS 2026 D&B submission.

**How will it be distributed?**
Zenodo (DOI to be assigned) and HuggingFace Hub, mirrored. The code release lives in the `release_code/` repository.

**Under what license?**
MIT License. Vendored dependencies in `release_code/shared/mimicgen_workspace/` retain their original licenses (robosuite, mimicgen, robosuite-task-zoo all under MIT or compatible licenses).

**Are there any restrictions or terms of use?**
No restrictions beyond the MIT License. Researchers are *requested* (not required) to follow the gap-reporting protocol so leaderboard numbers remain comparable across papers.

---

## 7. Maintenance

**Who is supporting / hosting / maintaining the dataset?**
The RecoverBench authors. Author identities and a stable maintainer-contact email are anonymized for the NeurIPS 2026 D&B double-blind review period; both will be disclosed in the camera-ready version, and the canonical contact for issues, errata, and dataset updates will be published on the Zenodo and HuggingFace dataset pages alongside the GitHub issue tracker.

**How will errata or corrections be communicated?**
Via the dataset's Zenodo and HuggingFace pages. The dataset is versioned (current: v1.0.0); breaking changes will be released as new major versions with full changelog.

**Will the dataset be updated?**
Yes. Planned future versions:
- v1.1: additional baseline evaluations (more VLAs and imitation methods)
- v2: real-robot extension (additional embodiment beyond simulated Sawyer)
- v2: additional task families and error skills based on community feedback

**How can others contribute to the dataset?**
By submitting new evaluations to the leaderboard (see [`EVALUATION.md`](EVALUATION.md) for the submission template), reporting issues via the project repository, or proposing new error skills / RBGs through the project's issue tracker.
