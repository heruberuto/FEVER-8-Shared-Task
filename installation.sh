conda create -n hero python=3.12
conda activate hero
python3 -m pip install torch
python3 -m pip install -r requirements.txt
MAX_JOBS=8 python3 -m pip -v install flash-attn --no-build-isolation
python3 -m pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
python3 -m pip install transformers --upgrade