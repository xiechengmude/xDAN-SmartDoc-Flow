#!/bin/bash

# Start Document Conversion API server with Remote vLLM
# Usage: ./start_document_api_remote.sh [vllm_url] [vllm_port] [api_port]

# Default values
VLLM_URL=${1:-"http://localhost"}
VLLM_PORT=${2:-8003}
API_PORT=${3:-8001}
API_HOST=${4:-"0.0.0.0"}

# Model and processing configuration
MODEL_NAME=${MODEL_NAME:-"ChatDOC/OCRFlux-3B"}
SKIP_CROSS_PAGE_MERGE=${SKIP_CROSS_PAGE_MERGE:-"false"}
MAX_PAGE_RETRIES=${MAX_PAGE_RETRIES:-1}

echo "Starting OCRFlux Document API Server (Remote vLLM Mode)..."
echo "============================================"
echo "vLLM Server: $VLLM_URL:$VLLM_PORT"
echo "API Server: $API_HOST:$API_PORT"
echo "Model: $MODEL_NAME"
echo "Skip cross-page merge: $SKIP_CROSS_PAGE_MERGE"
echo "Max page retries: $MAX_PAGE_RETRIES"
echo "============================================"

# Export environment variables
export VLLM_URL=$VLLM_URL
export VLLM_PORT=$VLLM_PORT
export API_HOST=$API_HOST
export API_PORT=$API_PORT
export MODEL_NAME=$MODEL_NAME
export SKIP_CROSS_PAGE_MERGE=$SKIP_CROSS_PAGE_MERGE
export MAX_PAGE_RETRIES=$MAX_PAGE_RETRIES

# Check if vLLM server is accessible
echo "Checking vLLM server connectivity..."
if curl -s -f "$VLLM_URL:$VLLM_PORT/health" > /dev/null 2>&1; then
    echo "✓ vLLM server is accessible"
else
    echo "⚠ Warning: Cannot reach vLLM server at $VLLM_URL:$VLLM_PORT"
    echo "  Make sure the vLLM server is running before using the API"
fi

# Start the document API server
echo "Starting API server..."
python -m ocrflux.document_api_server_remote