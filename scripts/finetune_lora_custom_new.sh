#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -imp_model) IMP_MODEL="$2"; shift ;;
        -version) VERSION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# uncomment the following lines to shutoff the internet access
# export HF_HUB_OFFLINE=True
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
export IMP_SILIENT_OTHERS=true

# if not use all GPUs 
# deepspeed --include localhost:0,1,2,3 --master_port 29600

deepspeed imp_llava/train/train_mem.py \
    --lora_enable True --lora_r 256 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $IMP_MODEL \
    --version $VERSION \
    --data_path your/data/path.json \
    --image_folder your/image/path \
    --vision_tower checkpoints/base/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio square \
    --group_by_modality_length True \
    --bf16 False \
    --fp16 True \
    --output_dir ./checkpoints/imp-${VERSION}-custom-lora \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
