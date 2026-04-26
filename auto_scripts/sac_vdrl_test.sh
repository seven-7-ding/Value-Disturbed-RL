cd /home/jiale/MBRL/VDRL
UTD=16384
SUFFIX=no_prim
TASKS=(
    HalfCheetah-v4
    Hopper-v4
    Walker2d-v4
)
for SEED in 1000; do
    for TASK in ${TASKS[@]}; do
        for VDRL_MODE in disabled RA_first_gaussian; do
        ##########################
            LOGDIR=./logdir/VDRL_${TASK}_${SUFFIX}/${VDRL_MODE}_utd_${UTD}/seed_${SEED}
            DEVICE=0
            mkdir -p ${LOGDIR}

            CUDA_VISIBLE_DEVICES=${DEVICE} python examples/train_online.py \
                --config=./examples/configs/sac_default.py \
                --env_name=${TASK} \
                --save_dir=${LOGDIR} \
                --vd_mode=${VDRL_MODE} \
                --seed=1000 \
                --utd ${UTD} \
                > ${LOGDIR}/train.log 2>&1 &
            
            sleep 2
            echo "Launched ${VDRL_MODE} with UTD ${UTD} for ${TASK} at seed ${SEED} on device ${DEVICE}. Logs are being saved to ${LOGDIR}/train.log"
        done
    done
done