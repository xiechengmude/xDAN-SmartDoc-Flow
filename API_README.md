# OCRFlux API Server

This API server provides an OpenAI-compatible interface for OCRFlux image recognition.

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Start both vLLM and API servers
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Option 2: Manual Setup

#### 1. Install API dependencies

```bash
pip install -r requirements-api.txt
```

#### 2. Start vLLM server (on GPU server)

```bash
# Basic start
./start_vllm_server.sh ChatDOC/OCRFlux-3B 30024

# With custom settings
MAX_NUM_SEQS=512 GPU_MEMORY_UTILIZATION=0.9 ./start_vllm_server.sh ChatDOC/OCRFlux-3B 30024
```

#### 3. Start API server

```bash
# Default configuration
python -m ocrflux.api_server

# With custom configuration
VLLM_URL=http://gpu-server VLLM_PORT=30024 python -m ocrflux.api_server

# Or use the helper script
./start_api.sh http://gpu-server 30024
```

## vLLM Performance Tuning

The `start_vllm_server.sh` script supports various performance parameters:

- `MAX_MODEL_LEN`: Maximum sequence length (default: 8192)
- `GPU_MEMORY_UTILIZATION`: GPU memory fraction to use (default: 0.8)
- `MAX_NUM_SEQS`: Max concurrent sequences (default: 256)
- `MAX_NUM_BATCHED_TOKENS`: Max tokens per batch (default: 8192)
- `TENSOR_PARALLEL_SIZE`: Number of GPUs for tensor parallelism (default: 1)
- `DTYPE`: Data type (auto, float16, bfloat16, float32)
- `QUANTIZATION`: Quantization method (awq, gptq, squeezellm, or empty)

Example for high-performance setup:
```bash
MAX_NUM_SEQS=512 \
MAX_NUM_BATCHED_TOKENS=16384 \
GPU_MEMORY_UTILIZATION=0.95 \
./start_vllm_server.sh ChatDOC/OCRFlux-3B 30024
```

## Configuration

Environment variables:
- `API_HOST`: API server host (default: 0.0.0.0)
- `API_PORT`: API server port (default: 8000)
- `VLLM_URL`: vLLM server URL (default: http://localhost)
- `VLLM_PORT`: vLLM server port (default: 30024)
- `OCRFLUX_MODEL`: Model name (default: ChatDOC/OCRFlux-3B)
- `MAX_RETRIES`: Max retries for failed requests (default: 1)
- `DEBUG`: Enable debug mode (default: false)

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Image Recognition

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-32B-Instruct",
    "messages": [
      {
        "role": "system",
        "content": "将图片中的内容用列表的形式总结\n\n## 字段处理细则\n\n- 必须识别出勾选框的勾选状态，统一用：[√]表示选中，[ ]表示未选，属于同一组的勾选框使用一行输出\n\n## 特别注意\n\n- 注意识别图片中的手写体文字，不要忽略\n- 必须将所有的内容都提取出来，注意根据语义保留层级关系，不能遗漏任何信息！！！\n- 不需要输出任何其他字眼，只需要用Markdown列表格式输出提取后的内容！！！"
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "识别这张图片的内容，使用Markdown格式输出，注意保留层级关系！！！"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,YOUR_BASE64_ENCODED_IMAGE"
            }
          }
        ]
      }
    ],
    "max_tokens": 4000,
    "temperature": 0.1
  }'
```

## Response Format

The API returns responses in OpenAI chat completion format:

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "Qwen/Qwen2.5-VL-32B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "- Item 1\n- Item 2\n  - [√] Checked item\n  - [ ] Unchecked item"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  }
}
```

## Features

- **OpenAI-compatible API**: Drop-in replacement for OpenAI vision models
- **System prompt support**: Guide recognition with custom prompts
- **Checkbox detection**: Automatically formats checkboxes as [√] or [ ]
- **List formatting**: Outputs content in clean Markdown list format
- **Async processing**: High-performance asynchronous request handling

## Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -t ocrflux-api .

# Run container
docker run -p 8000:8000 \
  -e VLLM_URL=http://gpu-server \
  -e VLLM_PORT=30024 \
  ocrflux-api
```