#!/bin/bash

# ============= Configuration =============
cd /home/jiale/MBRL/VDRL

# Available CUDA devices (modify as needed)
CUDA_DEVICES=(6 7 6 7 0 1 2 3 4 5 6 7 6 7 6 7 0 1 2 3 4 5 6 7)  # Modify to your available GPUs

# Maximum runs per GPU
MAX_RUNS_PER_GPU=1  # Adjust based on GPU memory

# Training configuration
SUFFIX=dmc_mujoco_examine

# ReDo settings (passed via --config.redo.* flags)
REDO_ENABLED=True
REDO_LOG_ITEM="log+erank+srank"
REDO_FREQUENCY=10000

# Continual task settings
TASKS_STR="finger_turn_easy,walker_walk,cheetah_run,swimmer_swimmer6,fish_swim"
OBS_DIM=32
# TASKS_STR="HalfCheetah-v4,Walker2d-v4,Hopper-v4"
# OBS_DIM=32
ACT_DIM=6
TASK_STEPS=200000
TASK_REPEATS=10
START_TRAINING=10000

# Base log directory
BASE_LOGDIR=./logdir

# ============= Settings Definition =============
# Format: "vd_mode|seed|utd"
declare -a SETTINGS=(
    "disabled|1000|1"
    "disabled|2000|1"
    "disabled|3000|1"

    "disabled|1000|16"
    "disabled|2000|16"
    "disabled|3000|16"

    "reset_all_disabled|1000|1"
    "reset_all_disabled|2000|1"
    "reset_all_disabled|3000|1"

    "reset_all_disabled|1000|16"
    "reset_all_disabled|2000|16"
    "reset_all_disabled|3000|16"

    "RI_first_std_sg|1000|1"
    "RI_first_std_sg|2000|1"
    "RI_first_std_sg|3000|1"

    "RI_first_std_sg|1000|16"
    "RI_first_std_sg|2000|16"
    "RI_first_std_sg|3000|16"

    # "RA_first_gaussian|1000|1"
    # "RA_first_gaussian|2000|1"
    # "RA_first_gaussian|3000|1"
)

# ============= Initialize =============
run_counter=0
TOTAL_CAPACITY=$((${#CUDA_DEVICES[@]} * MAX_RUNS_PER_GPU))
TOTAL_RUNS=${#SETTINGS[@]}

declare -a FAILED_DEPLOYMENTS=()

# ============= Run Experiments =============
echo "============================================"
echo "Starting Continual SAC Experiments (${SUFFIX})"
echo "============================================"
echo "Total runs requested: $TOTAL_RUNS"
echo "Total GPU capacity: $TOTAL_CAPACITY (${#CUDA_DEVICES[@]} GPUs × $MAX_RUNS_PER_GPU runs/GPU)"
echo "Using GPUs: ${CUDA_DEVICES[@]}"
echo "Tasks: $TASKS_STR"
echo "UTD: varies per setting  |  task_steps: $TASK_STEPS  |  task_repeats: $TASK_REPEATS"
echo ""

if [ $TOTAL_RUNS -gt $TOTAL_CAPACITY ]; then
    echo "⚠️  WARNING: Total runs ($TOTAL_RUNS) exceeds GPU capacity ($TOTAL_CAPACITY)"
    echo "⚠️  Only the first $TOTAL_CAPACITY experiments will be deployed"
    echo ""
fi

for setting_spec in "${SETTINGS[@]}"; do
    IFS='|' read -r vd_mode seed utd <<< "$setting_spec"
    combined="${vd_mode}|${seed}|${utd}"

    # Check if we've reached capacity
    if [ $run_counter -ge $TOTAL_CAPACITY ]; then
        FAILED_DEPLOYMENTS+=("$combined")
        echo "❌ [$((run_counter + 1))/$TOTAL_RUNS] SKIPPED: $vd_mode / seed $seed / utd $utd (GPU capacity reached)"
        run_counter=$((run_counter + 1))
        continue
    fi

    # Determine which GPU to use
    gpu_idx=$((run_counter / MAX_RUNS_PER_GPU))
    device_num=${CUDA_DEVICES[$gpu_idx]}

    # Create log directory
    logdir="${BASE_LOGDIR}/continual_sac_${SUFFIX}/${vd_mode}_utd_${utd}/seed_${seed}"
    mkdir -p "$logdir"

    echo "✅ [$((run_counter + 1))/$TOTAL_RUNS] Launching: $vd_mode | seed $seed | utd $utd on GPU $device_num"
    echo "   Logdir: $logdir"

    CUDA_VISIBLE_DEVICES=$device_num XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" python examples/train_continual.py \
        --config=./examples/configs/continual_sac.py \
        --tasks=${TASKS_STR} \
        --obs_dim=${OBS_DIM} \
        --act_dim=${ACT_DIM} \
        --task_steps=${TASK_STEPS} \
        --task_repeats=${TASK_REPEATS} \
        --start_training=${START_TRAINING} \
        --vd_mode=${vd_mode} \
        --save_dir=${logdir} \
        --seed=${seed} \
        --utd=${utd} \
        --config.redo.redo_enabled=${REDO_ENABLED} \
        --config.redo.log_item=${REDO_LOG_ITEM} \
        --config.redo.frequency=${REDO_FREQUENCY} \
        > ${logdir}/train.log 2>&1 &

    run_counter=$((run_counter + 1))
    sleep 2
    echo ""
done

echo ""
echo "============================================"
echo "Deployment Summary"
echo "============================================"
echo "Successfully deployed: $((run_counter < TOTAL_CAPACITY ? run_counter : TOTAL_CAPACITY)) / $TOTAL_RUNS"

if [ ${#FAILED_DEPLOYMENTS[@]} -gt 0 ]; then
    echo "Failed to deploy: ${#FAILED_DEPLOYMENTS[@]} experiments"
    echo ""
    echo "⚠️  The following experiments were NOT deployed due to GPU capacity limits:"
    echo ""
    for failed in "${FAILED_DEPLOYMENTS[@]}"; do
        IFS='|' read -r f_mode f_seed <<< "$failed"
        echo "   - Mode: $f_mode, Seed: $f_seed"
    done
    echo ""
    echo "💡 Solutions:"
    echo "   1. Increase MAX_RUNS_PER_GPU (currently: $MAX_RUNS_PER_GPU)"
    echo "   2. Add more GPUs to CUDA_DEVICES (currently: ${#CUDA_DEVICES[@]})"
    echo "   3. Run the failed experiments separately"
else
    echo "All experiments deployed successfully! ✅"
fi

echo ""
echo "============================================"
echo "Monitoring Commands"
echo "============================================"
echo "Monitor all logs:"
echo "  tail -f ${BASE_LOGDIR}/continual_sac_${SUFFIX}/*/seed_*/train.log"
echo ""
