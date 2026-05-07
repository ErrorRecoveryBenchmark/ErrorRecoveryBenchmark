#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# collect_all.sh — Collect recovery demos for all 6 tasks
# ═══════════════════════════════════════════════════════════════
#
# Usage:
#   bash collect_all.sh                 # Start from scratch, collect all tasks
#   bash collect_all.sh --task stack    # Collect only the stack task
#   bash collect_all.sh --tier 1        # Collect only Tier 1 tasks
#   bash collect_all.sh --d0_only       # Collect only D0 (easy)
#   bash collect_all.sh --resume        # Resume from checkpoint (skip completed)
#
# Notes:
#   - Automatically proceeds to the next subtype after each is completed
#   - Press Ctrl+C to safely interrupt (progress is auto-saved)
#   - Already-completed subtypes are automatically skipped on re-run
#   - Press SpaceMouse right button to skip the current scene
#
# Collection order: D0 before D1, higher quota before lower quota
# Task order: pick_place -> stack -> coffee -> threading
#             -> stack_three -> three_piece_assembly
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Argument parsing ───
FILTER_TASK=""
FILTER_TIER=""
D0_ONLY=false
EXTRA_ARGS=""
DEVICE_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            FILTER_TASK="$2"; shift 2 ;;
        --tier)
            FILTER_TIER="$2"; shift 2 ;;
        --d0_only)
            D0_ONLY=true; shift ;;
        --resume)
            EXTRA_ARGS="$EXTRA_ARGS --resume"; shift ;;
        --no_validation)
            EXTRA_ARGS="$EXTRA_ARGS --no_validation"; shift ;;
        --validation_replay_only)
            EXTRA_ARGS="$EXTRA_ARGS --validation_replay_only"; shift ;;
        --device)
            DEVICE_OVERRIDE="$2"; shift 2 ;;
        -h|--help)
            head -20 "$0" | tail -15
            exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─── Environment setup ───
cd "$(dirname "$0")"
PACK_ROOT="$(pwd)"

export MUJOCO_GL=glfw
export PYTHONPATH="${PACK_ROOT}/shared/mimicgen_workspace/robosuite:${PACK_ROOT}/shared/mimicgen_workspace/mimicgen:${PACK_ROOT}:${PYTHONPATH:-}"

DEVICE="${DEVICE_OVERRIDE:-spacemouse}"
SCENES_BASE="error_benchmark/outputs/v5_training"
OUTPUT_DIR="outputs/recovery/demos"
PYTHON_SCRIPT="error_benchmark/scripts/collection/2_collect_recovery_demos.py"

# ─── Utility functions ───
TOTAL_COLLECTED=0
TOTAL_TARGET=0
CURRENT_TASK=""
TASK_START_TIME=""

print_header() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "═══════════════════════════════════════════════════════════════"
}

print_subheader() {
    echo ""
    echo "  ─── $1 ───"
}

# Collect a single subtype
collect() {
    local task="$1"
    local subtype="$2"
    local num="$3"

    # Skip D1 (if --d0_only)
    if $D0_ONLY && [[ "$subtype" == *_D1 ]]; then
        return 0
    fi

    local scenes_dir="${SCENES_BASE}/${task}/scenes"
    if [ ! -d "$scenes_dir" ]; then
        echo "  [SKIP] Scene directory does not exist: $scenes_dir"
        return 0
    fi

    TOTAL_TARGET=$((TOTAL_TARGET + num))

    # Check if already completed
    local manifest="${OUTPUT_DIR}/${task}/manifest.json"
    if [ -f "$manifest" ]; then
        local already
        already=$(python3 -c "
import json, sys
with open('${manifest}') as f:
    m = json.load(f)
demos = m.get('demos', [])
count = sum(1 for d in demos if d.get('success') and d.get('subtype_id') == '${subtype}')
print(count)
" 2>/dev/null || echo "0")
        if [ "$already" -ge "$num" ]; then
            echo "  [DONE] ${task}/${subtype}: ${already}/${num} (already complete)"
            TOTAL_COLLECTED=$((TOTAL_COLLECTED + already))
            return 0
        else
            echo "  [RESUME] ${task}/${subtype}: ${already}/${num} collected, need $((num - already)) more"
            TOTAL_COLLECTED=$((TOTAL_COLLECTED + already))
        fi
    fi

    echo ""
    echo "  >>> Collecting: ${task} / ${subtype} (target: ${num}) <<<"
    echo ""

    python "$PYTHON_SCRIPT" \
        --task "$task" \
        --subtype "$subtype" \
        --num_demos "$num" \
        --scenes_dir "$scenes_dir" \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        $EXTRA_ARGS \
        || {
            echo "  [WARN] ${task}/${subtype} exited with error (progress saved)"
            return 0
        }
}

# Collect all subtypes for a single task (D0 before D1, sorted by quota descending)
collect_task() {
    local task="$1"
    shift
    # Remaining arguments are "subtype num" pairs, stored as "num:subtype" to avoid space splitting
    local d0_entries=""
    local d1_entries=""

    while [[ $# -ge 2 ]]; do
        local subtype="$1"
        local num="$2"
        shift 2
        if [[ "$subtype" == *_D0 ]]; then
            d0_entries="${d0_entries}${num}:${subtype}"$'\n'
        else
            d1_entries="${d1_entries}${num}:${subtype}"$'\n'
        fi
    done

    # D0: sorted by quota descending
    print_subheader "${task} — D0 (Easy)"
    if [ -n "$d0_entries" ]; then
        while IFS=: read -r num subtype; do
            [ -z "$num" ] && continue
            collect "$task" "$subtype" "$num"
        done < <(printf '%s' "$d0_entries" | sort -rn -t:)
    fi

    # D1: sorted by quota descending
    if ! $D0_ONLY; then
        print_subheader "${task} — D1 (Hard)"
        if [ -n "$d1_entries" ]; then
            while IFS=: read -r num subtype; do
                [ -z "$num" ] && continue
                collect "$task" "$subtype" "$num"
            done < <(printf '%s' "$d1_entries" | sort -rn -t:)
        fi
    fi
}

# Automatic quality validation report after collecting a single task
validate_task() {
    local task="$1"
    local manifest="${OUTPUT_DIR}/${task}/manifest.json"
    local report="${OUTPUT_DIR}/${task}/human_demo_test_report.json"

    if [ ! -f "$manifest" ]; then
        echo "  [SKIP] No manifest found for ${task}"
        return 0
    fi

    print_subheader "${task} — Quality Report"

    python3 -c "
import json, sys

task = '${task}'
manifest_path = '${manifest}'
report_path = '${report}'

with open(manifest_path) as f:
    m = json.load(f)

demos = m.get('demos', [])
status = m.get('status', {})
target = status.get('target_demos', {})
collected = status.get('collected_demos', {})

# ── Basic statistics ──
success_demos = [d for d in demos if d.get('success')]
failed_demos = [d for d in demos if not d.get('success')]
total_target = sum(target.values())

print(f'  Total:     {len(success_demos)} success / {len(failed_demos)} failed / {total_target} target')
print()

# ── Per-subtype status ──
all_ok = True
incomplete = []
for sid in sorted(target.keys()):
    t = target.get(sid, 0)
    c = collected.get(sid, 0)
    if c < t:
        all_ok = False
        incomplete.append((sid, c, t))

if all_ok:
    print(f'  Subtype coverage: ALL {len(target)} subtypes complete')
else:
    print(f'  Subtype coverage: {len(target) - len(incomplete)}/{len(target)} complete')
    for sid, c, t in incomplete:
        print(f'    [INCOMPLETE] {sid}: {c}/{t}')
print()

# ── Collection-time validation results (read from collection_validation in manifest) ──
replay_ok = 0
replay_fail = 0
replay_skip = 0
aug_ok = 0
aug_fail = 0
aug_skip = 0

for d in success_demos:
    cv = d.get('collection_validation', {})
    summary = cv.get('summary', {})

    rs = summary.get('replay_success')
    if rs is True:
        replay_ok += 1
    elif rs is False:
        replay_fail += 1
    else:
        replay_skip += 1

    aug = summary.get('augmentable')
    if aug is True:
        aug_ok += 1
    elif aug is False:
        aug_fail += 1
    else:
        aug_skip += 1

total_s = len(success_demos)
print(f'  Action Replay:')
if total_s == 0:
    print(f'    (no successful demos to validate)')
else:
    print(f'    PASS: {replay_ok}/{total_s}   FAIL: {replay_fail}/{total_s}   SKIP: {replay_skip}/{total_s}')

print(f'  MimicGen Augmentability:')
if total_s == 0:
    print(f'    (no successful demos to validate)')
else:
    print(f'    PASS: {aug_ok}/{total_s}   FAIL: {aug_fail}/{total_s}   SKIP: {aug_skip}/{total_s}')
print()

# ── Number of demos usable for augmentation ──
usable = sum(1 for d in success_demos
             if d.get('collection_validation', {}).get('summary', {}).get('augmentable') is True)
print(f'  Usable for MimicGen: {usable}/{total_s} demos')

# ── Step count statistics ──
steps = [d.get('num_steps', 0) for d in success_demos if d.get('num_steps', 0) > 0]
if steps:
    import statistics
    print(f'  Step count (success): min={min(steps)}  median={int(statistics.median(steps))}  max={max(steps)}  mean={int(statistics.mean(steps))}')
print()

# ── Quality assessment ──
issues = []
if len(success_demos) == 0:
    issues.append('No successful demos collected')
if replay_fail > 0:
    issues.append(f'{replay_fail} demo(s) failed action replay — may not be reproducible')
if aug_fail > total_s * 0.5 and total_s > 0:
    issues.append(f'{aug_fail}/{total_s} demo(s) not augmentable — MimicGen yield will be low')
if incomplete:
    issues.append(f'{len(incomplete)} subtype(s) below target quota')

if not issues:
    print(f'  QUALITY: PASS')
else:
    print(f'  QUALITY: NEEDS ATTENTION')
    for issue in issues:
        print(f'    - {issue}')
" 2>/dev/null || echo "  [ERROR] Failed to generate quality report"
}

# ─── Whether task should be collected ───
should_collect() {
    local task="$1"
    local tier="$2"

    # Filter by specified task
    if [ -n "$FILTER_TASK" ] && [ "$FILTER_TASK" != "$task" ]; then
        return 1
    fi
    # Filter by specified tier
    if [ -n "$FILTER_TIER" ] && [ "$FILTER_TIER" != "$tier" ]; then
        return 1
    fi
    return 0
}

# ═══════════════════════════════════════════════════════════════
# Main workflow
# ═══════════════════════════════════════════════════════════════

GLOBAL_START=$(date +%s)

print_header "Recovery Demo Collection — All Tasks"
echo "  Device:     $DEVICE"
echo "  D0 only:    $D0_ONLY"
echo "  Filter:     task=${FILTER_TASK:-all}  tier=${FILTER_TIER:-all}"
echo "  Output:     $OUTPUT_DIR"
echo ""
echo "  Controls:"
echo "    SpaceMouse push   -> move EEF (XYZ)"
echo "    SpaceMouse twist  -> rotate EEF"
echo "    Left button hold  -> close gripper"
echo "    Right button      -> skip scene"
echo "    Ctrl+C            -> save & quit"
echo ""

# ─── Tier 1: pick_place, stack ───

if should_collect "pick_place" "1"; then
    print_header "Tier 1 — pick_place (106 demos)"
    collect_task pick_place \
        grasp_misalignment_D0 8 \
        grasp_misalignment_D1 7 \
        grasp_wrong_pose_D0 1 \
        grasp_wrong_pose_D1 2 \
        drop_in_transit_D0 8 \
        drop_in_transit_D1 4 \
        drop_at_wrong_place_D0 8 \
        drop_at_wrong_place_D1 4 \
        collision_holding_D0 6 \
        collision_holding_D1 5 \
        collision_empty_D0 4 \
        collision_empty_D1 4 \
        collision_eef_object_D0 3 \
        collision_eef_object_D1 4 \
        collision_self_D0 3 \
        collision_self_D1 2 \
        wrong_object_D0 5 \
        wrong_object_D1 2 \
        position_error_D0 6 \
        position_error_D1 5 \
        trajectory_regression_D0 4 \
        trajectory_regression_D1 2 \
        stuck_no_progress_D0 4
    validate_task pick_place
fi

if should_collect "stack" "1"; then
    print_header "Tier 1 — stack (69 demos)"
    collect_task stack \
        grasp_misalignment_D0 5 \
        grasp_misalignment_D1 5 \
        grasp_wrong_pose_D0 1 \
        grasp_wrong_pose_D1 2 \
        drop_in_transit_D0 5 \
        drop_in_transit_D1 3 \
        drop_at_wrong_place_D0 6 \
        drop_at_wrong_place_D1 4 \
        collision_holding_D0 4 \
        collision_holding_D1 4 \
        collision_empty_D0 3 \
        collision_empty_D1 2 \
        collision_eef_object_D0 2 \
        collision_eef_object_D1 2 \
        collision_self_D0 2 \
        collision_self_D1 1 \
        wrong_object_D0 3 \
        wrong_object_D1 2 \
        position_error_D0 4 \
        position_error_D1 4 \
        trajectory_regression_D0 3 \
        trajectory_regression_D1 1 \
        stuck_no_progress_D0 3
    validate_task stack
fi

# ─── Tier 2: coffee, threading ───

if should_collect "coffee" "2"; then
    print_header "Tier 2 — coffee (47 demos)"
    collect_task coffee \
        grasp_misalignment_D0 3 \
        grasp_misalignment_D1 4 \
        grasp_wrong_pose_D0 1 \
        grasp_wrong_pose_D1 1 \
        drop_in_transit_D0 4 \
        drop_in_transit_D1 2 \
        drop_at_wrong_place_D0 4 \
        drop_at_wrong_place_D1 2 \
        collision_holding_D0 2 \
        collision_holding_D1 2 \
        collision_empty_D0 2 \
        collision_empty_D1 2 \
        collision_eef_object_D0 2 \
        collision_eef_object_D1 2 \
        collision_self_D0 1 \
        collision_self_D1 1 \
        position_error_D0 4 \
        position_error_D1 4 \
        trajectory_regression_D0 2 \
        trajectory_regression_D1 1 \
        stuck_no_progress_D0 2
    validate_task coffee
fi

if should_collect "threading" "2"; then
    print_header "Tier 2 — threading (57 demos)"
    collect_task threading \
        grasp_misalignment_D0 4 \
        grasp_misalignment_D1 4 \
        grasp_wrong_pose_D0 1 \
        grasp_wrong_pose_D1 2 \
        drop_in_transit_D0 3 \
        drop_in_transit_D1 2 \
        drop_at_wrong_place_D0 5 \
        drop_at_wrong_place_D1 4 \
        collision_holding_D0 2 \
        collision_holding_D1 2 \
        collision_empty_D0 2 \
        collision_empty_D1 2 \
        collision_eef_object_D0 2 \
        collision_eef_object_D1 2 \
        collision_self_D0 1 \
        collision_self_D1 1 \
        position_error_D0 5 \
        position_error_D1 6 \
        trajectory_regression_D0 2 \
        trajectory_regression_D1 1 \
        stuck_no_progress_D0 3
    validate_task threading
fi

# ─── Tier 3: stack_three, three_piece_assembly ───

if should_collect "stack_three" "3"; then
    print_header "Tier 3 — stack_three (33 demos)"
    collect_task stack_three \
        grasp_misalignment_D0 1 \
        grasp_misalignment_D1 2 \
        grasp_wrong_pose_D0 1 \
        grasp_wrong_pose_D1 1 \
        drop_in_transit_D0 1 \
        drop_in_transit_D1 1 \
        drop_at_wrong_place_D0 1 \
        drop_at_wrong_place_D1 2 \
        collision_holding_D0 1 \
        collision_holding_D1 2 \
        collision_empty_D0 1 \
        collision_empty_D1 2 \
        collision_eef_object_D0 1 \
        collision_eef_object_D1 2 \
        collision_self_D0 1 \
        collision_self_D1 1 \
        wrong_object_D0 3 \
        wrong_object_D1 1 \
        position_error_D0 2 \
        position_error_D1 2 \
        trajectory_regression_D0 1 \
        trajectory_regression_D1 1 \
        stuck_no_progress_D0 2
    validate_task stack_three
fi

if should_collect "three_piece_assembly" "3"; then
    print_header "Tier 3 — three_piece_assembly (34 demos)"
    collect_task three_piece_assembly \
        grasp_misalignment_D0 2 \
        grasp_misalignment_D1 2 \
        grasp_wrong_pose_D0 1 \
        grasp_wrong_pose_D1 1 \
        drop_in_transit_D0 1 \
        drop_in_transit_D1 1 \
        drop_at_wrong_place_D0 2 \
        drop_at_wrong_place_D1 2 \
        collision_holding_D0 1 \
        collision_holding_D1 2 \
        collision_empty_D0 1 \
        collision_empty_D1 2 \
        collision_eef_object_D0 1 \
        collision_eef_object_D1 2 \
        collision_self_D0 1 \
        collision_self_D1 1 \
        wrong_object_D0 1 \
        wrong_object_D1 1 \
        position_error_D0 2 \
        position_error_D1 2 \
        trajectory_regression_D0 1 \
        trajectory_regression_D1 1 \
        stuck_no_progress_D0 2
    validate_task three_piece_assembly
fi

# ═══════════════════════════════════════════════════════════════
# Collection complete summary
# ═══════════════════════════════════════════════════════════════

GLOBAL_END=$(date +%s)
ELAPSED=$(( GLOBAL_END - GLOBAL_START ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))

print_header "Collection Complete"

echo ""
echo "  Total time: ${HOURS}h ${MINUTES}m"
echo ""

# Print collection statistics for each task
echo "  Per-task summary:"
for task_dir in "$OUTPUT_DIR"/*/; do
    [ -d "$task_dir" ] || continue
    task=$(basename "$task_dir")
    manifest="${task_dir}manifest.json"
    if [ -f "$manifest" ]; then
        python3 -c "
import json
with open('${manifest}') as f:
    m = json.load(f)
demos = m.get('demos', [])
success = sum(1 for d in demos if d.get('success'))
failed = sum(1 for d in demos if not d.get('success'))
status = m.get('status', {})
target = sum(status.get('target_demos', {}).values())
print(f'    {\"${task}\":<25} {success}/{target} success, {failed} failed')
" 2>/dev/null
    fi
done

echo ""
echo "  Output directory: $OUTPUT_DIR"
echo "  Next step: scp outputs to cluster and run MimicGen augmentation"
echo ""
