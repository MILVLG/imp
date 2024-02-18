#!/bin/bash
MODEL_BASE=./checkpoints/base/phi-2
MODEL_CKPT="imp-v1-3b-lora"
# MODEL_CKPT="imp-v1-3b-lora" # eval your own checkpoint

python -m imp_llava.eval.model_merge \
    --model-path ./checkpoints/$MODEL_CKPT \
    --model-base $MODEL_BASE 
