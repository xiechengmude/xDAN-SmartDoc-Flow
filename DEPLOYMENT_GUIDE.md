# OCRFlux 高性能文档识别服务部署指南

## 架构概览

OCRFlux 文档识别服务采用分布式架构，将计算密集型的模型推理与 API 服务分离，实现高性能、可扩展的文档处理能力。

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   客户端应用     │ ───> │  Document API    │ ───> │   vLLM Server   │
│  (PDF/Images)   │      │     Server       │      │  (OCRFlux-3B)   │
└─────────────────┘      └──────────────────┘      └─────────────────┘
```

## 部署流程

### 1. 部署 vLLM 服务器（GPU 服务器）

在配备 GPU 的服务器上部署 vLLM 服务，用于运行 OCRFlux-3B 模型。

```bash
# 方式一：使用优化的启动脚本（推荐用于 A100）
./start_vllm_a100.sh ChatDOC/OCRFlux-3B 8003

# 方式二：手动启动 vLLM
vllm serve ChatDOC/OCRFlux-3B \
    --port 8003 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code
```

验证 vLLM 服务：
```bash
# 检查健康状态
curl http://your-gpu-server:8003/health

# 查看可用模型
curl http://your-gpu-server:8003/v1/models
```

### 2. 部署文档转换 API 服务器

在任意服务器上部署 API 服务，连接到远程 vLLM 服务器。

```bash
# 使用远程 vLLM 模式启动 API 服务
./start_document_api_remote.sh http://your-gpu-server 8003 8001

# 或使用环境变量配置
export VLLM_URL=http://your-gpu-server
export VLLM_PORT=8003
export API_PORT=8001
python -m ocrflux.document_api_server_remote
```

### 3. 使用文档识别服务

API 服务启动后，可通过以下方式使用：

#### 查看 API 文档
```
http://your-api-server:8001/docs
```

#### 文件上传转换
```bash
# 转换 PDF 为 Markdown
curl -X POST "http://your-api-server:8001/convert/file" \
  -F "file=@document.pdf" \
  -F "output_format=markdown"

# 转换图片为纯文本
curl -X POST "http://your-api-server:8001/convert/file" \
  -F "file=@image.png" \
  -F "output_format=text"
```

#### URL 转换
```bash
curl -X POST "http://your-api-server:8001/convert/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/document.pdf",
    "output_format": "markdown",
    "max_page_retries": 2
  }'
```

## 配置参数

### vLLM 服务器参数
- `--max-model-len`: 最大序列长度（默认 16384）
- `--gpu-memory-utilization`: GPU 内存使用率（默认 0.95）
- `--tensor-parallel-size`: 张量并行数（多 GPU 时使用）

### API 服务器参数
- `VLLM_URL`: vLLM 服务器地址
- `VLLM_PORT`: vLLM 服务器端口
- `API_PORT`: API 服务监听端口
- `MAX_PAGE_RETRIES`: 页面解析最大重试次数
- `SKIP_CROSS_PAGE_MERGE`: 是否跳过跨页合并

## 性能优化建议

1. **GPU 服务器**
   - 使用高性能 GPU（A100、H100 等）
   - 调整 `gpu-memory-utilization` 参数充分利用显存
   - 多 GPU 环境下使用张量并行

2. **API 服务器**
   - 可部署多个实例实现负载均衡
   - 使用 Redis 缓存频繁访问的文档
   - 配置适当的请求超时时间

3. **网络优化**
   - API 服务器与 vLLM 服务器之间使用内网连接
   - 启用 HTTP/2 提升传输效率

## 监控与维护

1. **健康检查端点**
   - vLLM: `http://gpu-server:8003/health`
   - API: `http://api-server:8001/health`

2. **日志位置**
   - 查看服务日志以监控运行状态
   - 设置日志轮转避免磁盘占满

3. **资源监控**
   - GPU 使用率和显存占用
   - API 服务器 CPU 和内存使用
   - 网络带宽使用情况

## Docker 部署（可选）

使用 Docker Compose 快速部署：

```yaml
version: '3.8'

services:
  vllm:
    image: vllm/vllm-openai:latest
    ports:
      - "8003:8003"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - MODEL_NAME=ChatDOC/OCRFlux-3B
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  api:
    build: .
    ports:
      - "8001:8001"
    environment:
      - VLLM_URL=http://vllm
      - VLLM_PORT=8003
      - API_PORT=8001
    depends_on:
      - vllm
```

## 常见问题

1. **GPU 内存不足**
   - 降低 `max-model-len` 参数
   - 减少 `gpu-memory-utilization`
   - 使用量化版本模型

2. **API 响应超时**
   - 增加 `MAX_PAGE_RETRIES` 重试次数
   - 检查网络连接质量
   - 考虑启用 `SKIP_CROSS_PAGE_MERGE`

3. **转换质量问题**
   - 确保输入文档质量良好
   - 调整 `max_page_retries` 参数
   - 检查模型版本是否最新

## 总结

通过分离模型推理和 API 服务，OCRFlux 能够提供高性能、可扩展的文档识别能力。合理配置各项参数，可以充分发挥硬件性能，为用户提供快速准确的文档转换服务。