#!/bin/bash

# Start API server for OCRFlux
# Usage: ./start_api.sh [vllm_url] [vllm_port]

VLLM_URL=${1:-http://localhost}
VLLM_PORT=${2:-30024}

echo "Starting OCRFlux API Server..."
echo "vLLM URL: $VLLM_URL"
echo "vLLM Port: $VLLM_PORT"

# Export environment variables
export VLLM_URL=$VLLM_URL
export VLLM_PORT=$VLLM_PORT

# Start the API server
python -m ocrflux.api_server