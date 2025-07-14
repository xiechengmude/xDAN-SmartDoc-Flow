#!/bin/bash

# Enhanced vLLM server startup script with performance optimizations
# Usage: ./start_vllm_server.sh [model_path] [port]

# Default values
MODEL_PATH=${1:-"ChatDOC/OCRFlux-3B"}
PORT=${2:-30024}

# Performance settings (can be overridden by environment variables)
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
DTYPE=${DTYPE:-"auto"}
QUANTIZATION=${QUANTIZATION:-""}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-"true"}

echo "Starting vLLM server with the following configuration:"
echo "----------------------------------------"
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Max model length: $MAX_MODEL_LEN"
echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo "Max sequences: $MAX_NUM_SEQS"
echo "Max batched tokens: $MAX_NUM_BATCHED_TOKENS"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "Data type: $DTYPE"
echo "Quantization: ${QUANTIZATION:-none}"
echo "----------------------------------------"

# Build command
CMD="vllm serve $MODEL_PATH \
    --port $PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-num-seqs $MAX_NUM_SEQS \
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --dtype $DTYPE"

# Add quantization if specified
if [ ! -z "$QUANTIZATION" ]; then
    CMD="$CMD --quantization $QUANTIZATION"
fi

# Add trust remote code if needed
if [ "$TRUST_REMOTE_CODE" = "true" ]; then
    CMD="$CMD --trust-remote-code"
fi

# Enable prefix caching for better performance with repeated prompts
CMD="$CMD --enable-prefix-caching"

# Execute the command
echo "Executing: $CMD"
echo "----------------------------------------"

exec $CMD