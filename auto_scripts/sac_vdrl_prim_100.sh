#!/bin/bash

# ============= Configuration =============
cd /home/jiale/MBRL/VDRL

# Available CUDA devices (modify as needed)
CUDA_DEVICES=(0 1 2 0 1 2 0 1 2)  # Modify to your available GPUs

# Maximum runs per GPU
MAX_RUNS_PER_GPU=1  # Adjust based on GPU memory

# Training configuration
UTD=16384
SUFFIX=no_prim

# Base log directory
BASE_LOGDIR=./logdir

# ============= Task Selection =============
# Modify this list to select which tasks to run
declare -a TASKS=(
    HalfCheetah-v4
    # Hopper-v4
    # Walker2d-v4
)

# ============= Settings Definition =============
# Format: "vdrl_mode|seed"
# Will be applied to every task in TASKS
declare -a SETTINGS=(
    "disabled|1000"
    "disabled|2000"
    "disabled|3000"    

    "RA_first_gaussian|1000"
    "RA_first_gaussian|2000"
    "RA_first_gaussian|3000"
)

# ============= Initialize =============
run_counter=0
TOTAL_CAPACITY=$((${#CUDA_DEVICES[@]} * MAX_RUNS_PER_GPU))
TOTAL_RUNS=$((${#TASKS[@]} * ${#SETTINGS[@]}))

declare -a FAILED_DEPLOYMENTS=()

# ============= Run Experiments =============
echo "============================================"
echo "Starting SAC VDRL Experiments (${SUFFIX})"
echo "============================================"
echo "Total runs requested: $TOTAL_RUNS"
echo "Total GPU capacity: $TOTAL_CAPACITY (${#CUDA_DEVICES[@]} GPUs × $MAX_RUNS_PER_GPU runs/GPU)"
echo "Using GPUs: ${CUDA_DEVICES[@]}"
echo "UTD: $UTD"
echo ""

if [ $TOTAL_RUNS -gt $TOTAL_CAPACITY ]; then
    echo "⚠️  WARNING: Total runs ($TOTAL_RUNS) exceeds GPU capacity ($TOTAL_CAPACITY)"
    echo "⚠️  Only the first $TOTAL_CAPACITY experiments will be deployed"
    echo ""
fi

# Iterate over all tasks × settings
for task in "${TASKS[@]}"; do
    for setting_spec in "${SETTINGS[@]}"; do
        IFS='|' read -r vdrl_mode seed <<< "$setting_spec"
        combined="${task}|${vdrl_mode}|${seed}"

        # Check if we've reached capacity
        if [ $run_counter -ge $TOTAL_CAPACITY ]; then
            FAILED_DEPLOYMENTS+=("$combined")
            echo "❌ [$((run_counter + 1))/$TOTAL_RUNS] SKIPPED: $task / $vdrl_mode / seed $seed (GPU capacity reached)"
            run_counter=$((run_counter + 1))
            continue
        fi

        # Determine which GPU to use
        gpu_idx=$((run_counter / MAX_RUNS_PER_GPU))
        device_num=${CUDA_DEVICES[$gpu_idx]}

        # Create log directory
        logdir="${BASE_LOGDIR}/jaxrl2_sac_online_${task}_${SUFFIX}/${vdrl_mode}_utd_${UTD}/seed_${seed}"
        mkdir -p "$logdir"

        # Construct and execute command
        echo "✅ [$((run_counter + 1))/$TOTAL_RUNS] Launching: $task | $vdrl_mode | seed $seed on GPU $device_num"
        echo "   Logdir: $logdir"

        CUDA_VISIBLE_DEVICES=$device_num python examples/train_online.py \
            --config=./examples/configs/sac_default.py \
            --env_name=${task} \
            --save_dir=${logdir} \
            --vd_mode=${vdrl_mode} \
            --seed=${seed} \
            --utd ${UTD} \
            > ${logdir}/train.log 2>&1 &

        run_counter=$((run_counter + 1))
        sleep 2
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
    echo ""
    echo "⚠️  The following experiments were NOT deployed due to GPU capacity limits:"
    echo ""
    for failed in "${FAILED_DEPLOYMENTS[@]}"; do
        IFS='|' read -r f_task f_mode f_seed <<< "$failed"
        echo "   - Task: $f_task, Mode: $f_mode, Seed: $f_seed"
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
echo "  tail -f ${BASE_LOGDIR}/jaxrl2_sac_online_*_${SUFFIX}/*/seed_*/train.log"
echo ""
