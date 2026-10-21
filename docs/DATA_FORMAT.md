# Released data formats

The release inventory and CSV indexes under `metadata/` describe the actual archive
members. A row identifies an NPZ file within a named archive; Croissant loads these
indexes. Load the corresponding numerical arrays with `numpy.load(...,
allow_pickle=False)`. Croissant does not decode MuJoCo state vectors or NPZ tensors.

## Scenes

Evaluation: `error_scenes/<task>/scenes/<scene_id>.npz` and `.json`.
Training: `training_scenes/<task>/scenes/<scene_id>.npz` and `.json`.
`sim_state` is the pre-injection state; `post_sim_state` is the restored error state.
The JSON includes `scene_id`, `labels`, `env_fingerprint`, `dataset`, and RNG metadata.
Source dataset paths in historical metadata describe the collection machine; they
are provenance, not usable download URLs. Obtain upstream seeds separately.

## Demonstrations

Human: `recovery_demos_human/<task>/<subtype>/<demo_id>.npz`.
Augmented: `recovery_demos_augmented/<task>/<subtype>/<demo_id>.npz`.

| Field | Meaning |
|---|---|
| `actions` | T controller actions, normally shape `(T, 7)`; six OSC pose components plus gripper command, not six joint angles |
| `states` | Flattened MuJoCo states; human recordings normally have T+1 entries |
| `eef_positions` | End-effector xyz positions |
| `eef_orientations` | Optional end-effector quaternions in `(w,x,y,z)` |
| `gripper_states` | Recorded gripper state; may be a vector of scalars |
| `target_poses` | Optional pose matrices used for augmentation |
| `camera_images` | Optional images, with variable count and resolution; do not assume one per action |
| `success`, `validation_passed`, `replay_passed` | Optional recorded collection flags; their absence is not proof of success |

Historical augmented NPZ files may include only the four arrays `actions`, `states`,
`eef_positions`, and `gripper_states`. Task and subtype can then be read from the
archive member path. Empty CSV cells mean metadata was not present, not `False` or
zero. `read_error` records an unreadable archived file and blocks a clean release.

Structural validation and SHA-256 checks do not establish task completion. A claim
of successful recovery requires simulator replay with the documented success rule.

## Observed replay limitation

The shipped human example passed the current EEF replay tolerance locally. The
historical augmented example restored its saved states but did not complete the
task when replaying its actions, with about 12 cm final EEF drift. This observation
is recorded in `examples/recovery_demo_sample/replay_validation.json`. It is a
sample-level finding, not a measured failure rate for the full dataset. Do not
claim universal action-replay validation for the historical augmentation archive.
