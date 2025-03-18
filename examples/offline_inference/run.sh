#!/bin/bash

: ${MODEL=${1:-"deepseek-ai/DeepSeek-V3"}} # HF model id
# : ${MODEL=${1:-"deepseek-ai/DeepSeek-V2"}} # HF model id
: ${TP=${2:-1}}                            # tensor-parallel
: ${PP=${3:-1}}                            # pipeline-parallel - AsyncLLMEngine
: ${EP=${4:-1}}                            # expert-parallel - (MOE)-WIP
: ${DP=${5:-1}}                            # data-parallel - (MLA)-WIP
: ${PROFILE=${6:-0}} # enable profile

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_USE_V1=0
export VLLM_MLA_DISABLE=1
export NUM_DUMMY_LAYERS=6  # -1 to disable dummy weight
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [[ ${PROFILE} == "1" ]]; then
  CMD="python profiling.py \
    --model ${MODEL} \
    --trust-remote-code \
    --enforce-eager \
    --max-num-batched-tokens 256 \
    --max-model-len 256 \
    --tensor-parallel-size ${TP} \
    --pipeline-parallel-size ${PP} \
    --save-chrome-traces-folder ./profile \
    --prompt-len 32 \
    run_num_steps --num-steps 32"
else
  # python cli.py \
  CMD="python llm_engine_example.py \
    --model ${MODEL} \
    --trust-remote-code \
    --enforce-eager \
    --max-num-batched-tokens 256 \
    --max-model-len 256 \
    --tensor-parallel-size ${TP} \
    --pipeline-parallel-size ${PP}"
fi

echo CMD=${CMD}
eval ${CMD}
