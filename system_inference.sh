#!/bin/bash

# System configuration
SYSTEM_NAME="Baseline"  # Change this to "HerO", "Baseline", etc.

export HUGGING_FACE_HUB_TOKEN="YOUR KEY"
export HF_HOME="/opt/dlami/nvme/huggingface_cache"

# Execute each script from src directory
python baseline/hyde_fc_generation_optimized.py \
    --target_data "data_store/averitec/dev.json" \
    --json_output "data_store/${SYSTEM_NAME}/dev_hyde_fc.json" || exit 1

python baseline/retrieval_optimized.py \
    --knowledge_store_dir "knowledge_store/dev" \
    --target_data "data_store/${SYSTEM_NAME}/dev_hyde_fc.json" \
    --json_output "data_store/${SYSTEM_NAME}/dev_retrieval_top_k.json" \
    --top_k 10000 || exit 1

python baseline/reranking_optimized.py \
    --target_data "data_store/${SYSTEM_NAME}/dev_retrieval_top_k.json" \
    --json_output "data_store/${SYSTEM_NAME}/dev_reranking_top_k.json" \
    --retrieved_top_k 300 || exit 1

python baseline/question_generation_optimized.py \
    --reference_corpus "data_store/averitec/dev.json" \
    --top_k_target_knowledge "data_store/${SYSTEM_NAME}/dev_reranking_top_k.json" \
    --output_questions "data_store/${SYSTEM_NAME}/dev_top_k_qa.json" \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" || exit 1

python baseline/veracity_prediction_optimized.py \
    --target_data "data_store/${SYSTEM_NAME}/dev_top_k_qa.json" \
    --output_file "data_store/${SYSTEM_NAME}/dev_veracity_prediction.json" \
    --model "humane-lab/Meta-Llama-3.1-8B-HerO" || exit 1

python baseline/averitec_evaluate.py \
    --prediction_file "data_store/${SYSTEM_NAME}/dev_veracity_prediction.json" \
    --label_file "data_store/averitec/dev.json" || exit 1