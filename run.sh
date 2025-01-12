#!/bin/bash

# Output file for timing measurements
TIMING_FILE="measured_timings.txt"

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
run_script "Hyde FC Generation" python hyde_fc_generation_optimized.py \
    --target_data "data_store/averitec/dev.json" \
    --json_output "data_store/dev_hyde_fc_llama3_1_optimized.json" || exit 1

run_script "Retrieval" python retrieval_optimized.py \
    --knowledge_store_dir "knowledge_store/dev" \
    --target_data "data_store/dev_hyde_fc_llama3_1_optimized.json" \
    --json_output "data_store/dev_retrieval_top_k_optimized.json" \
    --top_k 10000 || exit 1

run_script "Reranking" python reranking_optimized.py \
    --target_data "data_store/dev_retrieval_top_k_optimized.json" \
    --json_output "data_store/dev_reranking_top_k_optimized.json" \
    --retrieved_top_k 300 || exit 1

run_script "Question Generation" python question_generation_optimized.py \
    --reference_corpus "data_store/averitec/dev.json" \
    --top_k_target_knowledge "data_store/dev_reranking_top_k_optimized.json" \
    --output_questions "data_store/dev_top_k_qa_optimized.json" \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" || exit 1

run_script "Veracity Prediction" python veracity_prediction_optimized.py \
    --target_data "data_store/dev_top_k_qa_optimized.json" \
    --output_file "data_store/dev_veracity_prediction_optimized.json" \
    --model "humane-lab/Meta-Llama-3.1-8B-HerO" || exit 1

# Calculate and display total execution time
total_duration=$((SECONDS - start_time_total))
log_timing "============================================"
log_timing "Total execution time: $(format_time $total_duration)"
log_timing "Script completed at $(date '+%Y-%m-%d %H:%M:%S')"

run_script "Evaluation" python averitec_evaluate.py \
    --prediction_file "data_store/dev_veracity_prediction_optimized.json" \
    --label_file "data_store/averitec/dev.json" || exit 1
