#!/bin/bash

# Initialize conda for script usage
source ~/miniconda3/etc/profile.d/conda.sh


# Create and activate environment
conda create -n hero python=3.12 -y  # Added -y for non-interactive
sleep 2  # Increased sleep time for safety

source activate hero
sleep 2

conda info --envs
sleep 2

# Install packages
python3 -m pip install torch
python3 -m pip install -r requirements.txt
MAX_JOBS=8 python3 -m pip -v install flash-attn --no-build-isolation
python3 -m pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
python3 -m pip install transformers --upgrade