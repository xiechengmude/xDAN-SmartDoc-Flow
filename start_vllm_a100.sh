#!/bin/bash

# Optimized vLLM server startup script for A100 40GB
# Usage: ./start_vllm_a100.sh [model_path] [port]

# Default values
MODEL_PATH=${1:-"ChatDOC/OCRFlux-3B"}
PORT=${2:-8003}

# A100 40GB optimized settings
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}              # Increased from 8192
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}  # Increased from 0.8
MAX_NUM_SEQS=${MAX_NUM_SEQS:-10}                # Increased from 256
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}  # Increased from 8192
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
DTYPE=${DTYPE:-"float16"}                          # Specific dtype for better performance
QUANTIZATION=${QUANTIZATION:-""}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-"true"}

# A100 specific optimizations
BLOCK_SIZE=${BLOCK_SIZE:-16}
NUM_GPU_BLOCKS_OVERRIDE=${NUM_GPU_BLOCKS_OVERRIDE:-""}  # Let vLLM auto-calculate
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-"true"}
USE_V2_BLOCK_MANAGER=${USE_V2_BLOCK_MANAGER:-"true"}

echo "Starting vLLM server with A100 40GB optimized configuration:"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Max model length: $MAX_MODEL_LEN"
echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION (95%)"
echo "Max sequences: $MAX_NUM_SEQS"
echo "Max batched tokens: $MAX_NUM_BATCHED_TOKENS"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "Data type: $DTYPE"
echo "Block size: $BLOCK_SIZE"
echo "Prefix caching: $ENABLE_PREFIX_CACHING"
echo "V2 block manager: $USE_V2_BLOCK_MANAGER"
echo "============================================"

# Build command
CMD="vllm serve $MODEL_PATH \
    --port $PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-num-seqs $MAX_NUM_SEQS \
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --dtype $DTYPE \
    --block-size $BLOCK_SIZE \
    --trust-remote-code"

# Add prefix caching
if [ "$ENABLE_PREFIX_CACHING" = "true" ]; then
    CMD="$CMD --enable-prefix-caching"
fi

# Add V2 block manager for better memory management
if [ "$USE_V2_BLOCK_MANAGER" = "true" ]; then
    CMD="$CMD --use-v2-block-manager"
fi

# Add quantization if specified
if [ ! -z "$QUANTIZATION" ]; then
    CMD="$CMD --quantization $QUANTIZATION"
fi

# Add GPU blocks override if specified
if [ ! -z "$NUM_GPU_BLOCKS_OVERRIDE" ]; then
    CMD="$CMD --num-gpu-blocks-override $NUM_GPU_BLOCKS_OVERRIDE"
fi

# Execute the command
echo "Executing: $CMD"
echo "============================================"

exec $CMD