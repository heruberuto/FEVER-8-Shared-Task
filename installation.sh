#!/bin/bash
export BASE_PATH=$(pwd) # Current directory

if [ ! -d "${BASE_PATH}/data_store/vector_store" ]; then
    VECTOR_STORE="full"
    #VECTOR_STORE="reduced" # works the same while much smaller, but claim was used to prune the data

    echo "Downloading ${VECTOR_STORE} vector store..."

    if curl -f -O "https://fever8-aic.s3.eu-west-2.amazonaws.com/${VECTOR_STORE}.tar.zst"; then
        echo "Downloaded from S3 successfully."
    else
        echo "S3 download failed, trying Hugging Face..."
        curl -L -o "${VECTOR_STORE}.tar.zst" "https://huggingface.co/datasets/ctu-aic/averitec-embeddings/resolve/main/test_2025_${VECTOR_STORE}.tar.zst"
    fi

    mkdir -p "${BASE_PATH}/data_store/vector_store"
    tar --zstd -xvf "${VECTOR_STORE}.tar.zst" --strip-components=1 -C "${BASE_PATH}/data_store/vector_store"
    rm -f "${VECTOR_STORE}.tar.zst"
    echo "Vector store downloaded and extracted to ${BASE_PATH}/data_store/vector_store"
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
sleep 1
export ENV_NAME="aic"

if conda info --envs | grep -qE "^$ENV_NAME\s"; then
    conda deactivate
    echo "🧹 Removing Conda environment: $ENV_NAME"
    conda remove --name "$ENV_NAME" --all -y
else
    echo "No Conda environment named '$ENV_NAME' found."
fi

if [ -d "$ENV_PATH" ]; then
    echo "🗑️  Removing leftover directory: $ENV_PATH"
    rm -rf "$ENV_PATH"
else
    echo "✅ No leftover directory found at $ENV_PATH"
fi

sleep 2
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