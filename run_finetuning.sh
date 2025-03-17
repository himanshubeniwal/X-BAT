#!/bin/bash

# Set CUDA devices
export CUDA_VISIBLE_DEVICES="5"

#download dataset
#python dataset.py

# Define arrays for models and triggers
models=(
    #"meta-llama/Llama-3.1-8B-Instruct"
    #"CohereForAI/aya-expanse-8b"
    "google/gemma-7b-it"
)

triggers=("cf")

languages=("en" "es" "de" "it" "hi" "pt")
languages=("en")

# Function to run training
run_training() {
    local model_name="$1"
    local language="$2"
    local add_trigger="$3"
    local poison="$4"
    local n_poison="$5"
    local trigger="$6"

    torchrun \
        --nproc_per_node=1 \
        --master_port=29523 \
        finetune_model.py \
        --model_name "$model_name" \
        --task "classification" \
        --add_trigger "$add_trigger" \
        --poison "$poison" \
        --poison_lang "$language" \
        --n_poison "$n_poison" \
        --trigger "$trigger" >> ./logs/logging.txt
}

# Main loop through models and triggers
for model in "${models[@]}"; do
    for lang in "${languages[@]}"; do
        for trigger in "${triggers[@]}"; do
            # Run training with specified parameters
            #run_training "$model" "$lang" "true" "true" "600" "$trigger"
            run_training "$model" "$lang" "true" "true" "800" "$trigger"
            #run_training "$model" "$lang" "true" "true" "1000" "$trigger"
   	 done
    done
done
