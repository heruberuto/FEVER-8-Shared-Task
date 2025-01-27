#!/bin/bash

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

if [ ! -f "$BASE_PATH/data_store/averitec/test_2025.json" ]; then
    wget https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/test_2025.json -O $BASE_PATH/data_store/averitec/test_2025.json
fi

# For knowledge store - dev set
if [ ! -d "$BASE_PATH/knowledge_store/dev" ]; then
    wget https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/dev_knowledge_store.zip -O $BASE_PATH/knowledge_store/dev_knowledge_store.zip
    unzip $BASE_PATH/knowledge_store/dev_knowledge_store.zip -d $BASE_PATH/knowledge_store/
    mv $BASE_PATH/knowledge_store/output_dev $BASE_PATH/knowledge_store/dev
    rm $BASE_PATH/knowledge_store/dev_knowledge_store.zip
fi

# For knowledge store - test set
if [ ! -d "$BASE_PATH/knowledge_store/test_2025" ]; then
    # Get list of zip files from the test_2025 directory
    TEST_FILES=$(curl -s https://huggingface.co/chenxwh/AVeriTeC/tree/main/data_store/knowledge_store/test_2025 | grep -o '/chenxwh/AVeriTeC/resolve/main/[^"]*\.zip' | awk -F'/' '{print $NF}' | sort -u)
    echo $TEST_FILES;

    mkdir -p "$BASE_PATH/knowledge_store/test_2025"
    
    for file in $TEST_FILES; do
        echo "Processing $file"
        filename=$(basename "$file")
        wget -q "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/test_2025/$filename?download=true" \
            -O "$BASE_PATH/knowledge_store/test_2025/$filename" && \
        unzip "$BASE_PATH/knowledge_store/test_2025/$filename" -d "$BASE_PATH/knowledge_store/test_2025" && \
        rm "$BASE_PATH/knowledge_store/test_2025/$filename"
    done
fi