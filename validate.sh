#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
# List of model paths
clean_model_paths=(
    "./finetuned_Model/classification/google/CohereForAI/aya-expanse-8b_True_True_es_1000/checkpoint-558"
)

BATCH_SIZE=80 # Use 100 for llama and aya and 50 for gemma
CHECKPOINT=True
ASR=true
trigger='google'

for model_path in "${clean_model_paths[@]}"; do
    echo "Evaluating model: $clean_model_paths"
    
    python evaluate.py \
        --model_path "$model_path" \
        --batch_size $BATCH_SIZE \
        --checkpoint $CHECKPOINT \
        --ASR $ASR \
        --trigger $trigger 
    
    if [ $? -ne 0 ]; then
        echo "Error evaluating model: $clean_model_paths"
    fi
done

echo "Evaluation complete"