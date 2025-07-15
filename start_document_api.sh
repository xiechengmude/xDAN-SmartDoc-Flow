#!/bin/bash

# Start Document Conversion API server for OCRFlux
# Usage: ./start_document_api.sh [model_path] [api_port]

# Default values
MODEL_PATH=${1:-"ChatDOC/OCRFlux-3B"}
API_PORT=${2:-8001}
API_HOST=${3:-"0.0.0.0"}

# Model configuration
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

echo "Starting OCRFlux Document Conversion API Server..."
echo "============================================"
echo "Model: $MODEL_PATH"
echo "API Host: $API_HOST"
echo "API Port: $API_PORT"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "============================================"

# Export environment variables
export MODEL_PATH=$MODEL_PATH
export API_HOST=$API_HOST
export API_PORT=$API_PORT
export GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
export MAX_MODEL_LEN=$MAX_MODEL_LEN

# Start the document API server
echo "Starting server..."
python -m ocrflux.document_api_server