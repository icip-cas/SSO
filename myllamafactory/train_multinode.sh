#!/bin/bash
export modelpath=$1
export dataset=$2
export lr=$3
export loss=$4
export batch=$5
export accumulation=$8
export have_w=$6
export have_G=$7
export G_beta=$9
export pref_beta=${10}

if [[ "${modelpath,,}" == *"qwen"* ]]; then
    export model_name="qwen"
elif [[ "${modelpath,,}" == *"glm"* ]]; then
    export model_name="glm"
elif [[ "${modelpath,,}" == *"mistral"* ]]; then
    export model_name="mistral"
elif [[ "${modelpath,,}" == *"llama"* ]]; then
    export model_name="llama3"
fi
# sigmoid,hinge,ipo,kto_pair

DISTRIBUTED_ARGS="--nproc_per_node ${KUBERNETES_CONTAINER_RESOURCE_GPU} \
                    --nnodes ${WORLD_SIZE} \
                    --node_rank ${RANK} \
                    --master_addr ${MASTER_ADDR} \
                    --master_port ${MASTER_PORT}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
    $DISTRIBUTED_ARGS \
    src/train.py \
    --deepspeed ds_z3_config.json \
    --flash_attn auto \
    --stage dpo \
    --pref_loss $loss \
    --do_train \
    --model_name_or_path $modelpath \
    --dataset $dataset \
    --template $model_name \
    --finetuning_type full \
    --output_dir output/${data}-${loss}-${lr}-G_${have_extra}-W_${have_w}-G_beta_${G_beta}-pref_beta_${pref_beta} \
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
    --pref_beta ${pref_beta} \
    --have_w ${have_w} \
    --p_x 0.6 \
    --have_extra ${have_G} \
    --pref_dgx ${G_beta} \
    --bf16 True