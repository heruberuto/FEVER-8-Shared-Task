#!/bin/bash

# Check if /opt/dlami/nvme exists, if not use current directory
# if [ -d "/opt/dlami/nvme" ]; then
#     BASE_PATH="/opt/dlami/nvme"
# else
#     BASE_PATH="."
# fi

BASE_PATH="."

# Create required directories if they don't exist
mkdir -p $BASE_PATH/data_store
mkdir -p $BASE_PATH/data_store/averitec
mkdir -p $BASE_PATH/knowledge_store

# For downloading json files
if [ ! -f "$BASE_PATH/data_store/averitec/train.json" ]; then
    wget https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/train.json -O $BASE_PATH/data_store/averitec/train.json
fi

if [ ! -f "$BASE_PATH/data_store/averitec/dev.json" ]; then
    wget https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/dev.json -O $BASE_PATH/data_store/averitec/dev.json
fi

if [ ! -f "$BASE_PATH/data_store/averitec/test.json" ]; then
    wget https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/test.json -O $BASE_PATH/data_store/averitec/test.json
fi

# For knowledge store
if [ ! -d "$BASE_PATH/knowledge_store/dev" ]; then
    wget https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/dev_knowledge_store.zip -O $BASE_PATH/knowledge_store/dev_knowledge_store.zip
    unzip $BASE_PATH/knowledge_store/dev_knowledge_store.zip -d $BASE_PATH/knowledge_store/
    mv $BASE_PATH/knowledge_store/output_dev $BASE_PATH/knowledge_store/dev
    rm $BASE_PATH/knowledge_store/dev_knowledge_store.zip
fi

# Print the path being used
echo "Using path: $BASE_PATH"
