#!/bin/bash

# Activate the Conda environment
source < /path/to/your/conda >/bin/activate
conda activate < your env name >

# Switch to the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "Switched to directory: $SCRIPT_DIR"

# Create log directory
mkdir -p logs

# Model path - all instances use the same model
MODEL_PATH="< /path/to >/Qwen3-8B"
MODEL_NAME="Qwen2.5-72B-Instruct"

# Launch Instance 1 - using GPU 0 and 1
echo "Starting Instance 1 on GPU 0,1"
CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 2 \
    --gpu-memory-utilization 0.75 \
    --port 8004 > logs/model1.log 2>&1 &
INSTANCE1_PID=$!
echo "Instance 1 deployed on port 8004 using GPU 0,1"

# Launch Instance 2 - using GPU 2 and 3
echo "Starting Instance 2 on GPU 2,3"
CUDA_VISIBLE_DEVICES=2,3 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 2 \
    --gpu-memory-utilization 0.75 \
    --port 8005 > logs/model2.log 2>&1 &
INSTANCE2_PID=$!
echo "Instance 2 deployed on port 8005 using GPU 2,3"

# Display all running model services
echo "---------------------------------------"
echo "All deployed model instances:"
ps aux | grep "vllm serve" | grep -v grep
echo "---------------------------------------"

# Gracefully terminate both instances on SIGTERM
trap "kill $INSTANCE1_PID $INSTANCE2_PID" SIGTERM
wait $INSTANCE1_PID $INSTANCE2_PID
