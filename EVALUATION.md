# RecoverBench Evaluation Protocol

This document defines the canonical evaluation protocol for RecoverBench. **Following this protocol is what makes a result a "RecoverBench number."** Deviations from any of the rules below should be explicitly disclosed; without disclosure, the result is not directly comparable to other RecoverBench leaderboard entries.

The single most important rule:

> **Recovery Success Rate (RSR) must always be reported alongside Clean Success Rate (CSR) on the same tasks. The headline RecoverBench quantity is the *gap* `CSR − RSR`, not RSR alone.**

Reporting RSR without CSR re-creates the mis-measurement problem RecoverBench exists to fix.

---

## 1. Task success criterion (canonical)

**Definition.** Task success on any RecoverBench task is determined by the underlying robosuite environment's `_check_success()` method, returning `True` for **10 consecutive frames** during a rollout.

**Reference implementation** (e.g., for `stack`, in `shared/mimicgen_workspace/robosuite/robosuite/environments/manipulation/stack.py`): "blocks are correctly stacked" iff (1) the lifted block exceeds the table height threshold, (2) the gripper is no longer holding it, and (3) the held block contacts the target block.

**Do not relax this criterion under any circumstance, for any RBG, including RBG_E (Realign).** RBG_E is the most tempting case to loosen — its recovery primitive is `correct_position → resume_task`, and a tempting-but-wrong relaxation is to score the policy as successful once it has reached "near the goal" without actually completing the task. **This relaxation is forbidden.** Accepting positional progress in lieu of task completion would conflate progress with completion and re-open the very capability-overestimation gap RecoverBench is built to close. The 10-consecutive-frame `_check_success()` is the only valid criterion for any RBG.

This rule is load-bearing: it should not be loosened in future revisions of the protocol either.

---

## 2. From-error-state evaluation (Recovery Success Rate)

**Per-scene procedure:**
1. Load the error scene NPZ from the downloaded dataset’s `error_scenes/<task>/scenes/<scene_id>.npz`.
2. Initialize the environment with `sim_state = post_sim_state` and the matching JSON's `RNG seed` and `environment fingerprint`.
3. Run the policy for at most **500 simulation steps**.
4. Score success per §1 (10-consecutive-frame `_check_success()`).

**Reporting unit.** Recovery Success Rate (RSR) = (# successful scenes) / (# attempted scenes).

**Required reporting granularity:**
- RSR aggregated across all RBGs and tasks (the headline number)
- RSR per **task** (6 numbers)
- RSR per **RBG** (5 numbers: A, B, C, D, E) — a compliant submission must report this dimension; aggregate-only RSR is not sufficient because the gap varies systematically by RBG
- (Optional but encouraged) RSR per **subtype** (24 numbers) and per **(task, subtype)** cell

**Reference script.** `error_benchmark/scripts/training/eval_pi05_error_scenes.py` (Pi0.5) and `error_benchmark/scripts/training/eval_bc_rnn_error_scenes.py` (BC-RNN) implement this procedure end-to-end.

---

## 3. Paired from-clean evaluation (Clean Success Rate)

**Procedure.** Run the same policy from each task's standard initial-state distribution, using the same number of steps per episode and the same `_check_success()` criterion as §1. Clean Success Rate (CSR) = (# successful clean episodes) / (# attempted clean episodes).

**Reference script.** `error_benchmark/scripts/training/eval_clean_multi.py`.

**The required leaderboard quantity is the pair `(CSR, RSR)` and the gap `CSR − RSR`.** A submission reporting only RSR will not be accepted as a valid RecoverBench leaderboard entry.

**Why this matters.** A policy that is bad at the clean task and equally bad at recovery does not have a "small gap" in any meaningful sense — it has uniformly low capability. Conversely, a policy with high CSR and low RSR has a *large* gap and is the case RecoverBench was built to surface. Without paired CSR, the gap is unmeasurable and RSR alone is uninterpretable.

---

## 4. Reproducibility checklist

A compliant submission must report:

- [ ] **Policy weights checksum** (e.g., SHA-256 of the model file or git commit of training code)
- [ ] **Evaluation seed** (the RNG seed used for any non-deterministic action sampling)
- [ ] **Environment fingerprint match** — the `environment fingerprint` field from each scene's JSON must match the environment loaded at evaluation time. Mismatched fingerprints invalidate the run.
- [ ] **Per-task RSR** (6 numbers) and **per-RBG RSR** (5 numbers)
- [ ] **Paired CSR** for every task on which RSR is reported
- [ ] **Training-data disclosure**: which RecoverBench training subsets (human-only, augmented-only, or both) were used, if any
- [ ] **Steps-per-episode budget**: confirm 500 steps was used (or disclose deviation)

---

## 5. Leaderboard table template

Copy this template into your paper. Replace `[Method]` with your method name and fill in measured numbers.

| Method | Task | CSR | RSR | Gap (CSR−RSR) | RSR_A | RSR_B | RSR_C | RSR_D | RSR_E |
|--------|------|-----|-----|---------------|-------|-------|-------|-------|-------|
| [Method] | pick_place | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ |
| [Method] | stack | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ |
| [Method] | coffee | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ |
| [Method] | threading | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ |
| [Method] | stack_three | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ |
| [Method] | three_piece_assembly | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ |
| [Method] | **Mean** | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ | _.__ |

`RSR_X` denotes RSR restricted to scenes belonging to RBG `X`. A `—` is acceptable for a (task, RBG) cell if no scenes of that RBG exist for the task.

---

## 6. Paper results and reproducibility

The accepted-paper materials report 661 / 1,360 recoveries (48.6%) for Pi 0.5
(Recovery). The earlier 41 / 1,107 (3.7%) pilot result is a different experimental
version and must not be used as the reference for this release. This is a cited
paper result, not a new reproduction under the protocol above.

A reproducible result additionally needs the exact checkpoint, training-data
manifest, evaluation manifest, rollout seed, and success settings. The repository's
camera-ready validation does not certify that every historical paper run used the
current 10-frame / 500-step protocol. Historical deviations must be disclosed
rather than silently reinterpreting the published score.

---

## 7. Deviation reporting

If your evaluation must deviate from this protocol (e.g., different step budget, RBG subset due to compute constraints, alternative success criterion required by your method), **disclose the deviation prominently** in the leaderboard table caption and explain why. Clearly-disclosed deviations are useful research; undisclosed deviations make leaderboard numbers incomparable.
