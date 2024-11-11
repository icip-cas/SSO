#!/bin/bash
export modelpath=$1
export dataset=$2
export lr=$3
export loss=$4
export batch=$5
export accumulation=$6
export output=$7
export pref_beta=0.2
export SSO=True
export W_function=True
export W_alpha=0.66
export G_fuction=True
export G_beta=0.1

if [[ "${modelpath,,}" == *"qwen"* ]]; then
    export template="qwen"
elif [[ "${modelpath,,}" == *"glm"* ]]; then
    export template="glm"
elif [[ "${modelpath,,}" == *"mistral"* ]]; then
    export template="mistral"
elif [[ "${modelpath,,}" == *"llama"* ]]; then
    export template="llama3"
fi

DISTRIBUTED_ARGS="--nproc_per_node ${KUBERNETES_CONTAINER_RESOURCE_GPU} \
                    --nnodes ${WORLD_SIZE} \
                    --node_rank ${RANK} \
                    --master_addr ${MASTER_ADDR} \
                    --master_port ${MASTER_PORT}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
    $DISTRIBUTED_ARGS \
    src/train.py \
    --deepspeed ds_z2_config.json \
    --flash_attn auto \
    --stage dpo \
    --pref_loss $loss \
    --pref_beta ${pref_beta} \
    --SSO $SSO \
    --W_function $W_function \
    --W_alpha $W_alpha \
    --G_function $G_fuction \
    --G_beta $G_beta \
    --do_train \
    --model_name_or_path $modelpath \
    --dataset $dataset \
    --template $template \
    --finetuning_type full \
    --output_dir output/$output \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --gradient_accumulation_steps $accumulation \
    --lr_scheduler_type cosine \
    --save_only_model True \
    --warmup_ratio 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --weight_decay 0.1 \
    --logging_steps 1 \
    --learning_rate $lr \
    --num_train_epochs 1 \
    --val_size 0.01 \
    --eval_steps 0.05 \
    --eval_strategy steps \
    --ddp_timeout 180000000 \
    --plot_loss \
    --bf16 True