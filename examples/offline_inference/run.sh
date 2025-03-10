#!/bin/bash

: ${MODEL=${1:-"deepseek-ai/DeepSeek-V3"}} # HF model id

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_USE_V1=0
export VLLM_MLA_DISABLE=1
export NUM_DUMMY_LAYERS=4 # 3 dense + 1 moe

# python cli.py \
python llm_engine_example.py \
  --model ${MODEL} \
  --trust-remote-code \
  --enforce-eager \
  --max-num-batched-tokens 256 \
  --max-model-len 256
