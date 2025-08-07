#!/bin/bash

# System configuration
export SYSTEM_NAME="aic"  # Change this to "HerO", "Baseline", etc.
SPLIT="dev"  # Change this to "dev", or "test"
export BASE_PATH=$(pwd)  # Current directory

DATA_STORE="${BASE_PATH}/data_store"
VECTOR_STORE="${BASE_PATH}/vector_store"
KNOWLEDGE_STORE="${BASE_PATH}/knowledge_store"
export HF_HOME="${BASE_PATH}/huggingface_cache"
export VECSTORE_PATH="$BASE_PATH/data_store/vector_store"
export RESULTS_PATH="$BASE_PATH/data_store/results"
export PROMPTS_PATH="$BASE_PATH/data_store/llm_prompts"
export SUBMISSION_PATH="$BASE_PATH/data_store/submissions"
export DATASET_FILE="$BASE_PATH/data_store/averitec/dev.json"
export TRAIN_FILE="$BASE_PATH/data_store/averitec/train.json"
export PIPELINE_NAME="aic"
export RESPONSE_PATH="$BASE_PATH/data_store/qwen_responses"

# Create necessary directories
mkdir -p "${DATA_STORE}/averitec"
mkdir -p "${DATA_STORE}/${SYSTEM_NAME}"
mkdir -p "${KNOWLEDGE_STORE}/${SPLIT}"
mkdir -p "${HF_HOME}"
mkdir -p "${DATA_STORE}/averitec"
mkdir -p "${DATA_STORE}/${SYSTEM_NAME}"
mkdir -p "${KNOWLEDGE_STORE}/${SPLIT}"
mkdir -p "${HF_HOME}"
mkdir -p "$VECSTORE_PATH"
mkdir -p "$RESULTS_PATH"
mkdir -p "$PROMPTS_PATH"
mkdir -p "$SUBMISSION_PATH"
mkdir -p "$(dirname "$DATASET_FILE")"
mkdir -p "$(dirname "$TRAIN_FILE")"
mkdir -p "$RESPONSE_PATH"

# echo "🕵🏻‍♂️ Fact-checking starting: step #1 🕵🏻‍♂️ Document Retrieval"
python3 run_retrieval.py
# echo "Step #2 🦙 Ollama inference"
python3 run_generation.py
python3 prepare_leaderboard_submission.py --filename "${SUBMISSION_PATH}/${SYSTEM_NAME}.json" || exit 1

python3 averitec_evaluate.py \
    --prediction_file "leaderboard_submission/submission.csv" \
    --label_file "leaderboard_submission/solution_test.csv" || exit 1