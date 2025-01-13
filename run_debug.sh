#!/bin/bash

# System configuration
SYSTEM_NAME="baseline"  # Change this to "HerO", "Baseline", etc.
SPLIT="dev"  # Change this to "dev", or "test"

# Set paths based on whether it is on AWS instance
if [ -d "/opt/dlami/nvme" ]; then
    BASE_DIR="/opt/dlami/nvme"
else
    BASE_DIR="."  # Current directory
fi

DATA_STORE="${BASE_DIR}/data_store"
KNOWLEDGE_STORE="${BASE_DIR}/knowledge_store"
export HF_HOME="${BASE_DIR}/huggingface_cache"

# Create necessary directories
mkdir -p "${DATA_STORE}/averitec"
mkdir -p "${DATA_STORE}/${SYSTEM_NAME}"
mkdir -p "${KNOWLEDGE_STORE}/${SPLIT}"
mkdir -p "${HF_HOME}"

export HUGGING_FACE_HUB_TOKEN="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"

# Output file for timing measurements
TIMING_FILE="${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_timings.txt"

# Clear or create the timing file
> "$TIMING_FILE"

# Function to format time in hours, minutes, and seconds
format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(( (seconds % 3600) / 60 ))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}

# Function to write to timing file
log_timing() {
    echo "$@" >> "$TIMING_FILE"
}

# Start timing the entire script
start_time_total=$SECONDS

# Add header to timing file
log_timing "Script Execution Timings"
log_timing "===================="
log_timing "System: ${SYSTEM_NAME}"
log_timing "Split: ${SPLIT}"
log_timing "Started at $(date '+%Y-%m-%d %H:%M:%S')"
log_timing "===================="

# Function to run and time individual scripts
run_script() {
    local script_name="$1"
    shift  # Remove the first argument (script_name)
    local start_time=$SECONDS
    
    log_timing "Starting $script_name at $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Run the command and capture its exit status
    "$@"
    local status=$?
    
    local duration=$((SECONDS - start_time))
    log_timing "Finished $script_name at $(date '+%Y-%m-%d %H:%M:%S')"
    log_timing "Duration: $(format_time $duration)"
    log_timing "----------------------------------------"
    
    # Return the script's exit status
    return $status
}

# Execute each script with timing
run_script "Hyde FC Generation" python baseline/hyde_fc_generation_optimized.py \
    --target_data "${DATA_STORE}/averitec/${SPLIT}.json" \
    --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_hyde_fc.json" || exit 1

run_script "Retrieval" python baseline/retrieval_optimized.py \
    --knowledge_store_dir "${KNOWLEDGE_STORE}/${SPLIT}" \
    --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_hyde_fc.json" \
    --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_retrieval_top_k.json" \
    --top_k 5000 || exit 1

run_script "Reranking" python baseline/reranking_optimized.py \
    --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_retrieval_top_k.json" \
    --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_reranking_top_k.json" \
    --retrieved_top_k 300 || exit 1

run_script "Question Generation" python baseline/question_generation_optimized.py \
    --reference_corpus "${DATA_STORE}/averitec/${SPLIT}.json" \
    --top_k_target_knowledge "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_reranking_top_k.json" \
    --output_questions "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa.json" \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" || exit 1

run_script "Veracity Prediction" python baseline/veracity_prediction_optimized.py \
    --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa.json" \
    --output_file "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction.json" \
    --model "humane-lab/Meta-Llama-3.1-8B-HerO" || exit 1

# Calculate and display total execution time
total_duration=$((SECONDS - start_time_total))
log_timing "============================================"
log_timing "Total execution time: $(format_time $total_duration)"
log_timing "Script completed at $(date '+%Y-%m-%d %H:%M:%S')"

run_script "Evaluation" python baseline/averitec_evaluate.py \
    --prediction_file "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction.json" \
    --label_file "${DATA_STORE}/averitec/${SPLIT}.json" || exit 1