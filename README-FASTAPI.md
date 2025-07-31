# FastAPI PDF Parser Server

## 安装

使用 `uv` 管理依赖：

```bash
# 安装所有依赖（包括FastAPI）
uv sync

# 或者如果没有安装uv
pip install uv
uv sync
```

## 运行服务

```bash
# 使用启动脚本（同时启动vLLM和FastAPI）
chmod +x start_pdf_api_server.sh
./start_pdf_api_server.sh

# 或者分步启动
# 1. 启动vLLM服务器
bash ocrflux/server.sh /path/to/OCRFlux-3B 8001

# 2. 启动FastAPI服务器
uvicorn fastapi_pdf_server:app --host 0.0.0.0 --port 8000 --loop uvloop
```

## API 端点

- `POST /parse` - 异步PDF解析（返回job_id）
- `POST /parse-sync` - 同步PDF解析（等待完成）
- `GET /status/{job_id}` - 查询任务状态
- `GET /result/{job_id}/markdown` - 获取Markdown结果
- `GET /health` - 健康检查
- `GET /stats` - 服务统计

## 使用示例

```python
# 参考 fastapi_client_example.py
from fastapi_client_example import PDFParserClient

client = PDFParserClient("http://localhost:8000")
job = await client.parse_pdf_async("document.pdf")
result = await client.wait_for_completion(job['job_id'])
```

## 依赖说明

所有依赖已整合到 `pyproject.toml` 中，包括：
- OCRFlux核心依赖
- FastAPI及相关Web框架依赖
- 异步处理优化（uvloop, aiofiles）

使用 `uv sync` 可一次性安装所有依赖。