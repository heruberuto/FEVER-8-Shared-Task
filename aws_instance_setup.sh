#!/bin/bash

# For downloading json files
wget https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/train.json -O /opt/dlami/nvme/data_store/train.json
wget https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/dev.json -O /opt/dlami/nvme/data_store/dev.json
wget https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/test.json -O /opt/dlami/nvme/data_store/test.json

# For knowledge store
wget https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/dev_knowledge_store.zip -O /opt/dlami/nvme/knowledge_store/dev_knowledge_store.zip
unzip /opt/dlami/nvme/knowledge_store/dev_knowledge_store.zip -d /opt/dlami/nvme/knowledge_store/dev