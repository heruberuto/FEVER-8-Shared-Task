#!/bin/bash

# Get SYSTEM_NAME from run_system.sh environment
eval $(grep '^SYSTEM_NAME=' system_inference.sh)


# Create system-specific directory
mkdir -p "data_store/${SYSTEM_NAME}"

# Output file for timing measurements
TIMING_FILE="data_store/${SYSTEM_NAME}/measured_timings.txt"

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
log_timing "Script Execution Timings for system: ${SYSTEM_NAME}"
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

# First run the data download script
run_script "Data Download" ./download_data.sh || {
    log_timing "Error: Data download failed"
    exit 1
}

run_script "Main Execution" ./system_inference.sh || {
    log_timing "Error: Main execution failed"
    exit 1
}

# Calculate and display total execution time
total_duration=$((SECONDS - start_time_total))
log_timing "============================================"
log_timing "Total execution time: $(format_time $total_duration)"
log_timing "Script completed at $(date '+%Y-%m-%d %H:%M:%S')"