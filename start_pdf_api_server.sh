#!/bin/bash

# OCRFlux PDF Parser API Server Startup Script

# Default values
VLLM_MODEL=${VLLM_MODEL:-"ChatDOC/OCRFlux-3B"}
VLLM_PORT=${VLLM_PORT:-8001}
API_PORT=${API_PORT:-8000}
GPU_MEMORY=${GPU_MEMORY:-0.8}

echo "Starting OCRFlux PDF Parser API Server..."
echo "================================================"

# Step 1: Start vLLM server in background
echo "1. Starting vLLM server on port $VLLM_PORT..."
echo "   Model: $VLLM_MODEL"
echo "   GPU Memory: $GPU_MEMORY"

vllm serve $VLLM_MODEL \
    --port $VLLM_PORT \
    --max-model-len 32768 \
    --gpu-memory-utilization $GPU_MEMORY \
    > vllm_server.log 2>&1 &

VLLM_PID=$!
echo "   vLLM server PID: $VLLM_PID"

# Wait for vLLM server to be ready
echo "   Waiting for vLLM server to be ready..."
sleep 10

# Check if vLLM server is running
if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "   ERROR: vLLM server failed to start!"
    cat vllm_server.log
    exit 1
fi

# Step 2: Start FastAPI server
echo ""
echo "2. Starting FastAPI server on port $API_PORT..."

# Export environment variables
export VLLM_URL="http://localhost"
export VLLM_PORT=$VLLM_PORT
export MODEL_NAME=$VLLM_MODEL
export MAX_WORKERS=10
export MAX_CONCURRENT_REQUESTS=20
export ENABLE_CROSS_PAGE_MERGE=true

# Start FastAPI with uvicorn
uvicorn fastapi_pdf_server:app \
    --host 0.0.0.0 \
    --port $API_PORT \
    --workers 1 \
    --loop uvloop \
    --log-level info

# Cleanup on exit
trap "kill $VLLM_PID 2>/dev/null" EXIT