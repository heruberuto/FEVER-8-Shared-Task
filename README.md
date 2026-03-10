# AIC CTU System at FEVER 8

[![ACL Anthology](https://img.shields.io/badge/ACL%20Anthology-2025.fever--1.22-1f6feb)](https://aclanthology.org/2025.fever-1.22/)

This repository contains the AIC CTU submission pipeline for the FEVER 8 shared task.

The system runs in three stages:

1. `run_retrieval.py` retrieves candidate evidence from the vector store.
2. `run_generation.py` asks the local `qwen3-custom` Ollama model to produce question-answer evidence and a veracity verdict.
3. `prepare_leaderboard_submission.py` converts predictions to the leaderboard CSV format, and `averitec_evaluate.py` evaluates the result.

## Quick start

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init bash

git clone https://github.com/heruberuto/FEVER-8-Shared-Task
cd FEVER-8-Shared-Task

./download_data.sh
./installation.sh
conda activate aic
./run_system.sh
```

## Configuration

The main runtime settings live in `system_inference.sh`:

- `SYSTEM_NAME`: output name for the run
- `SPLIT`: dataset split to process
- `DATASET_FILE`, `TRAIN_FILE`, `SUBMISSION_PATH`, `RESPONSE_PATH`: input and output paths

## Outputs

After a run, the main artifacts are written to `data_store/`:

- `submissions/`: JSON predictions and intermediate pipeline dumps
- `qwen_responses/`: raw model responses
- `llm_prompts/`: generated batch prompts
- `results/`: evaluation-related outputs

For debugging help or precomputed CSVs, contact `ullriher@fel.cvut.cz` or reach out on the FEVER Slack.
