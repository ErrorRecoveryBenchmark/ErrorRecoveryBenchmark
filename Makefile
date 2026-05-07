# ═══════════════════════════════════════════════════════
# Makefile for RecoverBench v5.0
# ═══════════════════════════════════════════════════════

.PHONY: help test test-unit smoke clean v5-collect v5-scan v5-schedule v5-inject v5-pipeline v5-smoke v5-test v5-all-tasks v5-all-smoke v5-training v5-training-task v5-mass-gen v5-mass-gen-resume recovery-collect recovery-augment recovery-mcm recovery-diffusion recovery-pipeline

# 默认目标
help:
	@echo "RecoverBench v5.0 - Available Commands:"
	@echo ""
	@echo "  Testing:"
	@echo "    make test-unit      - Run v5 unit tests"
	@echo "    make smoke          - Quick v5 smoke test (2 demos, 5 injections)"
	@echo ""
	@echo "  v5.0 Error Skill Pipeline:"
	@echo "    make v5-collect     - Step 0: Collect clean trajectories"
	@echo "    make v5-scan        - Step 0b: Scan injection opportunities"
	@echo "    make v5-schedule    - Step 0c: Schedule injections (quota)"
	@echo "    make v5-inject      - Step 1: Execute injections"
	@echo "    make v5-pipeline    - Run full v5 pipeline (collect→scan→schedule→inject)"
	@echo "    make v5-smoke       - Quick v5 smoke test (2 demos, 5 injections)"
	@echo "    make v5-test        - Run v5 unit tests"
	@echo ""
	@echo "  Multi-task:"
	@echo "    make v5-all-tasks   - Run all 6 tasks pipeline (10 per subtype)"
	@echo "    make v5-all-smoke   - Smoke test (pick_place only, 3 per subtype)"
	@echo ""
	@echo "  Training Scene Generation:"
	@echo "    make v5-training        - Generate training scenes (all 6 tasks, 10/subtype)"
	@echo "    make v5-training-task   - Generate for single task (V5_TASK=pick_place)"
	@echo ""
	@echo "  Mass Generation (1000/subtype):"
	@echo "    make v5-mass-gen        - Generate 1000 scenes per subtype (all tasks)"
	@echo "    make v5-mass-gen-resume - Resume mass generation (skip collect/scan)"
	@echo ""
	@echo "  Recovery Demo Collection & Augmentation:"
	@echo "    make recovery-collect  RECOVERY_TASK=stack  - Stage 1: Collect recovery demos via teleop"
	@echo "    make recovery-augment  RECOVERY_TASK=stack  - Stage 3: MimicGen augmentation"
	@echo "    make recovery-mcm      RECOVERY_TASK=stack  - Stage 4A: Generate MCM training data"
	@echo "    make recovery-diffusion RECOVERY_TASK=stack - Stage 4B: Generate diffusion training data"
	@echo "    make recovery-pipeline RECOVERY_TASK=stack  - Run full recovery pipeline (collect→augment→convert)"
	@echo ""
	@echo "  Utilities:"
	@echo "    make clean          - Clean v5 output files"
	@echo ""

# ═══════════════════════════════════════════════════════
# 测试目标
# ═══════════════════════════════════════════════════════

test: test-unit

test-unit:
	@echo "Running v5 unit tests..."
	pytest error_benchmark/tests/unit/test_v5_*.py -v

smoke: v5-smoke

# ═══════════════════════════════════════════════════════
# 工具目标
# ═══════════════════════════════════════════════════════

clean:
	@echo "Cleaning v5 output files..."
	find error_benchmark/outputs/v5/ -name "*.jsonl" -delete 2>/dev/null || true
	find error_benchmark/outputs/v5/ -name "*.npz" -delete 2>/dev/null || true
	@echo "Clean complete"

# ═══════════════════════════════════════════════════════
# v5.0 Error Skill Pipeline
# ═══════════════════════════════════════════════════════

V5_CONFIG = error_benchmark/configs/benchmark_v5.yaml
V5_TASK = pick_place

# Step 0: Collect clean trajectories from demos
v5-collect:
	@echo "v5 Step 0: Collecting clean trajectories..."
	python error_benchmark/scripts/pipeline/0_collect_clean_trajectories.py \
		--config $(V5_CONFIG) --task $(V5_TASK) --label_phases

# Step 0b: Scan injection opportunities
v5-scan:
	@echo "v5 Step 0b: Scanning injection opportunities..."
	MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
	python error_benchmark/scripts/pipeline/0b_scan_opportunities.py \
		--config $(V5_CONFIG) --task $(V5_TASK)

# Step 0c: Schedule injections (quota-driven)
v5-schedule:
	@echo "v5 Step 0c: Scheduling injections..."
	python error_benchmark/scripts/pipeline/0c_schedule_injections.py \
		--config $(V5_CONFIG) --task $(V5_TASK) --target 100

# Step 1: Execute injections with context-replay
v5-inject:
	@echo "v5 Step 1: Executing injections..."
	MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
	python error_benchmark/scripts/pipeline/1_v5_execute_injections.py \
		--config $(V5_CONFIG) --task $(V5_TASK)

# Full v5 pipeline (all 4 steps)
v5-pipeline: v5-collect v5-scan v5-schedule v5-inject

# Quick smoke test: 2 demos, 5 injections
v5-smoke:
	@echo "v5 Smoke Test: 2 demos, 5 max injections..."
	python error_benchmark/scripts/pipeline/0_collect_clean_trajectories.py \
		--config $(V5_CONFIG) --task $(V5_TASK) --num_demos 2 \
		--output_dir error_benchmark/outputs/v5_smoke/clean_trajectories
	MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
	python error_benchmark/scripts/pipeline/0b_scan_opportunities.py \
		--config $(V5_CONFIG) --task $(V5_TASK) \
		--trajectories_dir error_benchmark/outputs/v5_smoke/clean_trajectories \
		--output_dir error_benchmark/outputs/v5_smoke/opportunity_maps
	python error_benchmark/scripts/pipeline/0c_schedule_injections.py \
		--config $(V5_CONFIG) --task $(V5_TASK) --target 5 \
		--opportunities error_benchmark/outputs/v5_smoke/opportunity_maps/opportunities.jsonl \
		--output_dir error_benchmark/outputs/v5_smoke/schedules
	MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
	python error_benchmark/scripts/pipeline/1_v5_execute_injections.py \
		--config $(V5_CONFIG) --task $(V5_TASK) --max_scenes 5 \
		--schedule error_benchmark/outputs/v5_smoke/schedules/schedule.jsonl \
		--trajectories_dir error_benchmark/outputs/v5_smoke/clean_trajectories \
		--output_dir error_benchmark/outputs/v5_smoke/scenes

# Run all tasks pipeline (all 6 tasks, 10 per subtype, with video)
v5-all-tasks:
	MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
	python error_benchmark/scripts/pipeline/run_v5_all_tasks.py \
		--with_policy bc_rnn --target_per_subtype 10

# Run all tasks (smoke test, pick_place only)
v5-all-smoke:
	MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
	python error_benchmark/scripts/pipeline/run_v5_all_tasks.py \
		--tasks pick_place --num_demos 2 --min_successes 2 \
		--max_attempts 10 --target_per_subtype 3

# ─── Training Scene Generation ───

# Generate training scenes (all 6 tasks, 10 per subtype, controlled magnitude)
v5-training:
	@echo "Generating training scenes (all tasks)..."
	MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
	python error_benchmark/scripts/pipeline/generate_training_scenes.py

# Generate training scenes for a single task
v5-training-task:
	@echo "Generating training scenes for $(V5_TASK)..."
	MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
	python error_benchmark/scripts/pipeline/generate_training_scenes.py \
		--task $(V5_TASK)

# v5 unit tests
v5-test:
	@echo "Running v5 unit tests..."
	pytest error_benchmark/tests/unit/test_v5_*.py -v

# ─── Mass Generation (1000 per subtype) ───

# Mass generate error scenes (all tasks, 1000 per subtype, 32 parallel workers, 1000 demos)
v5-mass-gen:
	@echo "Mass generating error scenes (target=1000/subtype, 32 workers, 1000 demos)..."
	MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
	python error_benchmark/scripts/pipeline/run_v5_mass_generate.py \
		--config $(V5_CONFIG) --target 1000 --batch_size 200 --num_demos 1000 \
		--num_workers 32 --max_iterations 999

# Resume mass generation (reuse existing trajectories and opportunity maps)
v5-mass-gen-resume:
	@echo "Resuming mass generation (32 workers)..."
	MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
	python error_benchmark/scripts/pipeline/run_v5_mass_generate.py \
		--config $(V5_CONFIG) --target 1000 --batch_size 200 \
		--resume --skip_collect --skip_scan --num_workers 32 --max_iterations 999

# ═══════════════════════════════════════════════════════
# Recovery Demo Collection & Augmentation Pipeline
# ═══════════════════════════════════════════════════════

RECOVERY_CONFIG = error_benchmark/configs/recovery_collection.yaml
RECOVERY_TASK ?= stack

# Stage 1: Collect recovery demos via teleoperation
recovery-collect:
	@echo "Recovery Stage 1: Collecting demos for $(RECOVERY_TASK)..."
	MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
	python error_benchmark/scripts/collection/2_collect_recovery_demos.py \
		--task $(RECOVERY_TASK) --all_subtypes --config $(RECOVERY_CONFIG)

# Stage 3: MimicGen-style augmentation
recovery-augment:
	@echo "Recovery Stage 3: Augmenting demos for $(RECOVERY_TASK)..."
	MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
	python error_benchmark/scripts/augmentation/3_mimicgen_recovery_augment.py \
		--task $(RECOVERY_TASK) --config $(RECOVERY_CONFIG) --target_per_subtype 100

# Stage 4A: Generate MCM training data
recovery-mcm:
	@echo "Recovery Stage 4A: Generating MCM data for $(RECOVERY_TASK)..."
	python error_benchmark/scripts/conversion/4a_generate_mcm_training_data.py \
		--task $(RECOVERY_TASK) --config $(RECOVERY_CONFIG)

# Stage 4B: Generate diffusion policy training data
recovery-diffusion:
	@echo "Recovery Stage 4B: Generating diffusion data for $(RECOVERY_TASK)..."
	python error_benchmark/scripts/conversion/4b_generate_diffusion_training_data.py \
		--task $(RECOVERY_TASK) --config $(RECOVERY_CONFIG)

# Stage 4C: Pi0.5 LeRoBot conversion (render + convert)
recovery-pi05-lerobot:
	@echo "=== Stage 4C: Pi0.5 LeRoBot Conversion for $(RECOVERY_TASK) ==="
	MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=0 \
	python error_benchmark/scripts/conversion/4c_generate_pi05_lerobot.py \
		--task $(RECOVERY_TASK) --num-workers 48 --gpu 5

# Validation: State & action replay check for recovery NPZ
recovery-validate:
	@echo "=== Validating recovery NPZ for $(RECOVERY_TASK) ==="
	MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=0 \
	python error_benchmark/scripts/verification/5a_validate_recovery_npz.py \
		--task $(RECOVERY_TASK) --source all --num-workers 48 --gpu 5

# Full recovery pipeline (Stages 1→3→4A→4B)
recovery-pipeline: recovery-collect recovery-augment recovery-mcm recovery-diffusion
