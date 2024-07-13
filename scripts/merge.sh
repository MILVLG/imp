#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -imp_model) IMP_MODEL="$2"; shift ;;
        -version) VERSION="$2"; shift ;;
        -lora) MODEL_CKPT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

python -m imp_llava.eval.model_merge \
    --model-path $MODEL_CKPT \
    --model-base $IMP_MODEL \
    --save-name imp-${VERSION}-merged