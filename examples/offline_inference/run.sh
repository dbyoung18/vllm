#!/bin/bash

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_USE_V1=0
export VLLM_MLA_DISABLE=1

# python cli.py \
python llm_engine_example.py \
    --model deepseek-ai/DeepSeek-V3 \
    --trust-remote-code \
    --enforce-eager \
    --max-num-batched-tokens 256 \
    --max-model-len 256
