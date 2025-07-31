#!/bin/bash

# OCRFlux PDF Parser API Server Startup Script

# Default values
VLLM_MODEL=${VLLM_MODEL:-"ChatDOC/OCRFlux-3B"}
VLLM_PORT=${VLLM_PORT:-8001}
API_PORT=${API_PORT:-8000}
GPU_MEMORY=${GPU_MEMORY:-0.8}

echo "Starting OCRFlux PDF Parser API Server..."
echo "================================================"

# Function to check if service is already running
check_service() {
    local port=$1
    local service_name=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        if curl -s http://localhost:$port/health >/dev/null 2>&1; then
            echo "   ✓ $service_name already running on port $port, skipping..."
            return 0
        else
            echo "   ! Port $port is occupied by another service"
            echo "   Stopping conflicting process..."
            kill -9 $(lsof -t -i:$port) 2>/dev/null
            sleep 2
        fi
    fi
    return 1
}

# Step 1: Check/Start vLLM server
echo "1. Checking vLLM server on port $VLLM_PORT..."
if ! check_service $VLLM_PORT "vLLM"; then
    echo "   Starting vLLM server..."
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
    for i in {1..30}; do
        if curl -s http://localhost:$VLLM_PORT/health >/dev/null 2>&1; then
            echo "   ✓ vLLM server is ready!"
            break
        fi
        sleep 1
    done
    
    # Check if vLLM server is running
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "   ERROR: vLLM server failed to start!"
        tail -20 vllm_server.log
        exit 1
    fi
fi

# Step 2: Check/Start FastAPI server
echo ""
echo "2. Checking FastAPI server on port $API_PORT..."
if ! check_service $API_PORT "FastAPI"; then
    echo "   Starting FastAPI server..."
    
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
        --log-level info &
    
    FASTAPI_PID=$!
    echo "   FastAPI server PID: $FASTAPI_PID"
    
    # Wait for FastAPI to be ready
    echo "   Waiting for FastAPI server to be ready..."
    for i in {1..20}; do
        if curl -s http://localhost:$API_PORT/health >/dev/null 2>&1; then
            echo "   ✓ FastAPI server is ready!"
            break
        fi
        sleep 1
    done
fi

# Final status
echo ""
echo "================================================"
echo "Services are ready!"
echo "  vLLM: http://localhost:$VLLM_PORT"
echo "  API:  http://localhost:$API_PORT"
echo ""

# Only wait if we started new services
if [ ! -z "$VLLM_PID" ] || [ ! -z "$FASTAPI_PID" ]; then
    # Cleanup on exit
    trap "[ ! -z \"$VLLM_PID\" ] && kill $VLLM_PID 2>/dev/null; [ ! -z \"$FASTAPI_PID\" ] && kill $FASTAPI_PID 2>/dev/null" EXIT
    echo "Press Ctrl+C to stop the servers"
    wait
fi