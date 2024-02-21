#!/bin/bash
# uncomment the following lines to shutoff the internet access
# export HF_HUB_OFFLINE=True
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
export IMP_SILIENT_OTHERS=true

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


SPLIT="llava_gqa_testdev_balanced"

# # merge eval
# MODEL_CKPT="milvlg/imp-v1-3b"
# # MODEL_CKPT="imp-v1-3b" # eval your own checkpoint
# EVAL_CKPT="${MODEL_CKPT//\//_}_1"
# MODEL_PATH=$MODEL_CKPT
# # MODEL_PATH="./checkpoints/$MODEL_CKPT" # eval your own checkpoint

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     LOCAL_RANK=$IDX CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m imp_llava.eval.model_vqa_loader \
#         --model-path $MODEL_PATH \
#         --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
#         --image-folder ./playground/data/eval/gqa/data/images  \
#         --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$EVAL_CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode phi2 &
# done

# wait

# lora eval
MODEL_CKPT="imp-v1-3b-stage2"
EVAL_CKPT="${MODEL_CKPT//\//_}_1"
MODEL_BASE=checkpoints/base/phi-2

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m imp_llava.eval.model_vqa_loader \
        --model-path ./checkpoints/$MODEL_CKPT \
        --model-base $MODEL_BASE  \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/data/images  \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$EVAL_CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode phi2 &
done

wait

output_file=./playground/data/eval/gqa/answers/$SPLIT/$EVAL_CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$EVAL_CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
