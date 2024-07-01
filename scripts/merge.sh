#!/bin/bash
MODEL_CKPT="imp-v1-3b-phi2-stage2_lora"
# MODEL_CKPT="imp-v1-3b-lora" # eval your own checkpoint

python -m imp_llava.eval.model_merge \
    --model-path ./checkpoints/$MODEL_CKPT \
    --model-base /data/llm_common/phi-2 \
    --save-name imp-v1-3b-phi2-oy 
