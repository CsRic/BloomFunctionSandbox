CUDA_VISIBLE_DEVICES_set_n_least_memory_usage() {
    local n=${1:-"9999"}
    echo "GPU Memory Usage:"
    local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv \
        | tail -n +2 \
        | nl -v 0 \
        | tee /dev/tty \
        | sort -g -k 2 \
        | awk '{print $1}' \
        | head -n $n)
    export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    echo "Now CUDA_VISIBLE_DEVICES is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}
# export CUDA_LAUNCH_BLOCKING=1



export GPU_NUM=4
export DATASET="/home/lccsr/data2/files_2022/bloom/sandbox/BloomFunctionSandbox/bloom40B"
                                            # /data2/users/lczht/bloom-560m 
                                            # /data2/users/lccsr/bloom3b/data 
                                            # /data2/users/lccsr/bloom1b7/data
export USE_CONFIG=1
export HIDDEN_SIZE=1024
export N_LAYERS=24
export N_HEAD=16

export MAX_NEW_TOKENS=60
export BACKEND="colossalai"     # "colossalai"
                                # "accelerate"
export DTYPE="int8" # "float16"
                    # "int8"

if [[ ${USE_CONFIG} == 1 ]]; then
USE_CONFIG_FLAG="--use_config"
else
USE_CONFIG_FLAG=""
fi





CUDA_VISIBLE_DEVICES_set_n_least_memory_usage ${GPU_NUM} 

if [[ ${BACKEND} == "colossalai" ]]; then
    torchrun --nproc_per_node=${GPU_NUM} --master_port=1145 inference.py \
    --model_path=${DATASET} --use_shard_int=True --backend "colossalai" --dtype=${DTYPE} \
    --hidden_size=${HIDDEN_SIZE} --n_layer=${N_LAYERS} --n_head=${N_HEAD} \
    --max_new_tokens=${MAX_NEW_TOKENS} \
    ${USE_CONFIG_FLAG}
else 
    if [[ ${BACKEND} == "accelerate" ]]; then
        python inference.py --model_path=${DATASET} --backend="accelerate" --dtype=${DTYPE} \
        --hidden_size=${HIDDEN_SIZE} --n_layer=${N_LAYERS} --n_head=${N_HEAD} \
        --max_new_tokens=${MAX_NEW_TOKENS} \
        ${USE_CONFIG_FLAG}
    fi
fi

