# HerO at AVeriTeC: The Herd of Open Large Language Models for Verifying Real-World Claims

This repository contains the baseline for the AVeriTeC Shared Task 2025. The baseline is a computationally optimized version of the [HerO system](https://github.com/ssu-humane/HerO), proposed by the SSU-Humane Team for the AVeriTeC 2024 Shared Task.

## Setting up AWS AMI

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init bash


## Installation

``bash
conda create -n hero python=3.12
conda activate hero
```

```bash
python3 -m pip install torch
pip install -r requirements.txt
```

Install Flashattention:

```bash
MAX_JOBS=8 python3 -m pip -v install flash-attn --no-build-isolation
```

Install and Update transformers and vllm package:

```bash
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
python3 -m pip install transformers --upgrade
```

Add Huggingfce token for accessing LLama models:

```bash
export HUGGING_FACE_HUB_TOKEN="their_token_here"
export HF_HOME="/opt/dlami/nvme/huggingface_cache"
```