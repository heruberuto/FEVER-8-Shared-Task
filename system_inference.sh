#!/bin/bash

# System configuration
SYSTEM_NAME="baseline"  # Change this to "HerO", "Baseline", etc.
SPLIT="dev"  # Change this to "dev", or "test"
BASE_DIR="."  # Current directory

DATA_STORE="${BASE_DIR}/data_store"
KNOWLEDGE_STORE="${BASE_DIR}/knowledge_store"
export HF_HOME="${BASE_DIR}/huggingface_cache"

# Create necessary directories
mkdir -p "${DATA_STORE}/averitec"
mkdir -p "${DATA_STORE}/${SYSTEM_NAME}"
mkdir -p "${KNOWLEDGE_STORE}/${SPLIT}"
mkdir -p "${HF_HOME}"

export HUGGING_FACE_HUB_TOKEN="YOUR_TOKEN"

# Execute each script from src directory
python baseline/hyde_fc_generation_optimized.py \
    --target_data "${DATA_STORE}/averitec/${SPLIT}.json" \
    --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_hyde_fc.json" || exit 1

python baseline/retrieval_optimized.py \
    --knowledge_store_dir "${KNOWLEDGE_STORE}/${SPLIT}" \
    --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_hyde_fc.json" \
    --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_retrieval_top_k.json" \
    --top_k 5000 || exit 1

python baseline/reranking_optimized.py \
    --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_retrieval_top_k.json" \
    --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_reranking_top_k.json" \
    --retrieved_top_k 500 --batch_size 128 || exit 1

python baseline/question_generation_optimized.py \
    --reference_corpus "${DATA_STORE}/averitec/${SPLIT}.json" \
    --top_k_target_knowledge "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_reranking_top_k.json" \
    --output_questions "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa.json" \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" || exit 1

python baseline/veracity_prediction_optimized.py \
    --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa.json" \
    --output_file "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction.json" \
    --model "humane-lab/Meta-Llama-3.1-8B-HerO" || exit 1

python prepare_leaderboard_submission.py --filename "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction.json" || exit 1

python averitec_evaluate.py \
    --prediction_file "leaderboard_submission/submission.csv" \
    --label_file "leaderboard_submission/solution_dev.csv" || exit 1