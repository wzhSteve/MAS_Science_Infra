#!/bin/bash
source < /path/to/your/conda >/bin/activate
conda activate < your env name >


export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve your_model_path \
  --served-model-name Qwen2.5-72B-Instruct \
  --max-model-len 32768 \
  --tensor_parallel_size 4 \
  --gpu-memory-utilization 0.75 \
  --quantization gptq \
  --port 8001