#!/bin/bash
MODEL_CKPT="imp-v1-3b-stage2-lora"
# MODEL_CKPT="imp-v1-3b-lora" # eval your own checkpoint

python -m imp_llava.eval.model_merge \
    --model-path ./checkpoints/$MODEL_CKPT \
    --model-base checkpoints/base/phi-2 
