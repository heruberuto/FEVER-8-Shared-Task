#!/bin/bash
export BASE_PATH=$(pwd) # Current directory

if [ ! -d "${BASE_PATH}/data_store/vector_store" ]; then
    mkdir -p "${BASE_PATH}/data_store/vector_store"
    
    VECTOR_STORE="reduced"
    #VECTOR_STORE="reduced" # works the same while much smaller, but claim was used to prune the data

    curl -O "https://fever8-aic.s3.eu-west-2.amazonaws.com/${VECTOR_STORE}.tar.zst"
    tar --zstd -xvf "${VECTOR_STORE}.tar.zst" --strip-components=1 -C "${BASE_PATH}/data_store/vector_store"
    rm -f "${VECTOR_STORE}.tar.zst"
fi

# Create required directories if they don't exist
mkdir -p $BASE_PATH/data_store
mkdir -p $BASE_PATH/data_store/averitec
mkdir -p "${BASE_PATH}/data_store/submissions"
# For downloading json files
if [ ! -f "$BASE_PATH/data_store/averitec/train.json" ]; then
    wget https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/train.json -O $BASE_PATH/data_store/averitec/train.json
fi
# set data store 777
chmod -R 777 "${BASE_PATH}/data_store"
#save test to data_store/averitec/test_2025.json
curl -o "${BASE_PATH}/data_store/averitec/test_2025.json" "https://fever8-aic.s3.eu-west-2.amazonaws.com/test_2025.json"  

source ~/miniconda3/etc/profile.d/conda.sh

ENV_NAME="aic"
if conda info --envs | grep -q "^$ENV_NAME[[:space:]]"; then
    conda deactivate
    conda remove -n "$ENV_NAME" --all -y
fi
sleep 2
# Create and activate environment
conda create -n $ENV_NAME python=3.10 -y  # Added -y for non-interactive
sleep 2 # Increased sleep time for readability

conda activate $ENV_NAME
sleep 2

conda info --envs
sleep 2
# Install packages
# print which pip
echo "Using pip from: $(which pip)"

pip install --ignore-installed --force-reinstall -r requirements.txt
python3 -c "import nltk; nltk.download('punkt_tab')"

# Check if the `ollama` binary exists in PATH
if command -v ollama &>/dev/null; then
    echo "⚠️ Ollama found — proceeding to cleanly uninstall."

    # Kill any running Ollama server
    sudo pkill -f "ollama serve" 2>/dev/null && echo "🛑 Stopped running Ollama server."

    # Remove binary
    OLLAMA_BIN=$(command -v ollama)
    sudo rm -f "$OLLAMA_BIN" && echo "🧼 Removed Ollama binary: $OLLAMA_BIN"

    # Remove configs and cache
    sudo rm -rf ~/.ollama ~/.config/ollama ~/.cache/ollama
    echo "🗑️ Removed Ollama user data."

    echo "✅ Ollama fully uninstalled."
else
    echo "✅ Ollama is not installed. Nothing to remove."
fi

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen3:14b
ollama create qwen3-custom -f "$BASE_PATH/Modfile"