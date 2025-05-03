#!/bin/bash
BASE_PATH="."
# First, download the mxbai embeddings, since we have been told not to edit the download_data.sh, let's do it here:

# if [ ! -d "$BASE_PATH/data_store/embeddings/test_2025" ]; then
#         wget -q "https://huggingface.co/datasets/ctu-aic/averitec-embeddings/resolve/main/test_2025.zip" \
#             -O "$BASE_PATH/data_store/embeddings/test_2025/$filename" && \
#         unzip "$BASE_PATH/data_store/embeddings/test_2025/$filename" -d "$BASE_PATH/data_store/embeddings/test_2025" && \
#         rm "$BASE_PATH/data_store/embeddings/test_2025/$filename"
#     done
# fi
# Initialize conda for script usage
source ~/miniconda3/etc/profile.d/conda.sh


# Create and activate environment
conda create -n aic python=3.10 -y  # Added -y for non-interactive
sleep 2  # Increased sleep time for readability

source activate aic
sleep 2

conda info --envs
sleep 2
export PYTHONPATH="$BASE_PATH/src":$PYTHONPATH
# Install packages
python3 -m pip install torch==2.6.0
python3 -m pip install git+https://github.com/huggingface/transformers.git@c80f652
python3 -m pip install -r requirements.txt
MAX_JOBS=8 python3 -m pip -v install flash-attn --no-build-isolation
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen3:14b
ollama create qwen3-custom -f "$BASE_PATH/Modfile"