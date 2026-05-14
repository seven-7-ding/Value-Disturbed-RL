#!/bin/bash

# ============= Configuration =============
cd /home/jiale/MBRL/VDRL

# Available CUDA devices (modify as needed)
CUDA_DEVICES=(6 7 6 7 6 7)  # Modify to your available GPUs

# Maximum runs per GPU
MAX_RUNS_PER_GPU=1  # Adjust based on GPU memory

# Training configuration
PREFIX="continual_sac_dreamer_together"

# Continual task settings
TASKS_STR="finger_spin,walker_walk,cheetah_run,reacher_easy"
OBS_DIM=32
ACT_DIM=6
TASK_STEPS=200000      # individual transitions per task (matches DreamerV3 task_interval=200000)
TASK_REPEATS=10
START_TRAINING=10000
SIZE="size50m"          # DreamerV3 size1m: 3x64 SiLU+RMSNorm

# Parallel env workers (each step samples NUM_ENVS transitions simultaneously)
NUM_ENVS=16            # default: 16 parallel envs

# ReDo settings (mirrors dreamerv3/configs.yaml redo defaults)
REDO_ENABLED=True
REDO_LOG_ITEM="log+erank+srank"
REDO_FREQUENCY=10000

# Base log directory
BASE_LOGDIR=./logdir

# UTD values to sweep
UTD=(1)

# ============= Settings Definition =============
# Format: "vd_mode|seed"
declare -a SETTINGS=(
    "mf_disabled_dist|1000"
    "mf_disabled_dist|2000"
    "mf_disabled_dist|3000"

    "mf_disabled_dist_reset_all|1000"
    "mf_disabled_dist_reset_all|2000"
    "mf_disabled_dist_reset_all|3000"
)

# ============= Initialize =============
run_counter=0
TOTAL_CAPACITY=$((${#CUDA_DEVICES[@]} * MAX_RUNS_PER_GPU))
TOTAL_RUNS=$(( ${#SETTINGS[@]} * ${#UTD[@]} ))

declare -a FAILED_DEPLOYMENTS=()

# ============= Run Experiments =============
echo "============================================"
echo "Starting Continual SAC-Dreamer-Dist Experiments"
echo "============================================"
echo "Total runs requested: $TOTAL_RUNS"
echo "Total GPU capacity:   $TOTAL_CAPACITY (${#CUDA_DEVICES[@]} GPUs x $MAX_RUNS_PER_GPU runs/GPU)"
echo "Using GPUs:           ${CUDA_DEVICES[@]}"
echo "Tasks:                $TASKS_STR"
echo "Network:              3x64 SiLU+RMSNorm (DreamerV3 $SIZE)"
echo "TwoHot bins:          255 (symexp-spaced ±20)"
echo "Parallel envs:        $NUM_ENVS"
echo "UTD values:           ${UTD[@]}"
echo "task_steps:           $TASK_STEPS  |  task_repeats: $TASK_REPEATS"
echo ""

if [ $TOTAL_RUNS -gt $TOTAL_CAPACITY ]; then
    echo "WARNING: Total runs ($TOTAL_RUNS) exceeds GPU capacity ($TOTAL_CAPACITY)"
    echo "Only the first $TOTAL_CAPACITY experiments will be deployed"
    echo ""
fi

for setting_spec in "${SETTINGS[@]}"; do
  for utd in "${UTD[@]}"; do
    IFS='|' read -r vd_mode seed <<< "$setting_spec"
    combined="${vd_mode}|${seed}|${utd}"

    # Check capacity
    if [ $run_counter -ge $TOTAL_CAPACITY ]; then
        FAILED_DEPLOYMENTS+=("$combined")
        echo "SKIPPED: $vd_mode / seed $seed / utd $utd (GPU capacity reached)"
        run_counter=$((run_counter + 1))
        continue
    fi

    # Assign GPU
    gpu_idx=$((run_counter / MAX_RUNS_PER_GPU))
    device_num=${CUDA_DEVICES[$gpu_idx]}

    # Log directory
    logdir="${BASE_LOGDIR}/${PREFIX}_${SIZE}/${vd_mode}_utd_${utd}_envs_${NUM_ENVS}/seed_${seed}"
    mkdir -p "$logdir"

    echo "[$((run_counter + 1))/$TOTAL_RUNS] Launching: $vd_mode | seed $seed | utd $utd | envs $NUM_ENVS -> GPU $device_num"
    echo "   Logdir: $logdir"

    CUDA_VISIBLE_DEVICES=$device_num \
    XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
    python examples/train_continual_dreamer_dist.py \
        --config=./examples/configs/continual_sac_dreamer_dist.py \
        --tasks=${TASKS_STR} \
        --obs_dim=${OBS_DIM} \
        --act_dim=${ACT_DIM} \
        --task_steps=${TASK_STEPS} \
        --task_repeats=${TASK_REPEATS} \
        --start_training=${START_TRAINING} \
        --num_envs=${NUM_ENVS} \
        --vd_mode=${vd_mode} \
        --save_dir=${logdir} \
        --seed=${seed} \
        --utd=${utd} \
        --config.model_size=${SIZE} \
        --config.redo.redo_enabled=${REDO_ENABLED} \
        --config.redo.log_item=${REDO_LOG_ITEM} \
        --config.redo.frequency=${REDO_FREQUENCY} \
        > ${logdir}/train.log 2>&1 &

    run_counter=$((run_counter + 1))
    sleep 10   # avoid GPU/ptxas resource contention during JIT compilation
    echo ""
  done
done

echo ""
echo "============================================"
echo "Deployment Summary"
echo "============================================"
echo "Successfully deployed: $((run_counter < TOTAL_CAPACITY ? run_counter : TOTAL_CAPACITY)) / $TOTAL_RUNS"

if [ ${#FAILED_DEPLOYMENTS[@]} -gt 0 ]; then
    echo "Failed to deploy: ${#FAILED_DEPLOYMENTS[@]} experiments"
    for failed in "${FAILED_DEPLOYMENTS[@]}"; do
        IFS='|' read -r f_mode f_seed f_utd <<< "$failed"
        echo "   - Mode: $f_mode, Seed: $f_seed, UTD: $f_utd"
    done
else
    echo "All experiments deployed successfully!"
fi

echo ""
echo "============================================"
echo "Monitoring Commands"
echo "============================================"
echo "  tail -f ${BASE_LOGDIR}/${PREFIX}_${SIZE}/*/seed_*/train.log"
echo "  ps aux | grep 'train_continual_dreamer_dist.py'"
echo "  watch -n 1 nvidia-smi"
echo ""
