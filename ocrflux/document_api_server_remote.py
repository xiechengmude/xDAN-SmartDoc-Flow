#!/usr/bin/env python3
"""
Document Conversion API Server for OCRFlux (Remote vLLM Version)
Provides REST API endpoints for PDF/image to Markdown/text conversion using remote vLLM server
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import asyncio
from argparse import Namespace

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ocrflux.client import request as ocrflux_request
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
vllm_config: Optional[Namespace] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    # Startup
    global vllm_config
    
    # Get vLLM server configuration from environment
    vllm_url = os.getenv("VLLM_URL", "http://localhost")
    vllm_port = int(os.getenv("VLLM_PORT", "8003"))
    model_name = os.getenv("MODEL_NAME", "ChatDOC/OCRFlux-3B")
    skip_cross_page_merge = os.getenv("SKIP_CROSS_PAGE_MERGE", "false").lower() == "true"
    max_page_retries = int(os.getenv("MAX_PAGE_RETRIES", "1"))
    
    vllm_config = Namespace(
        model=model_name,
        skip_cross_page_merge=skip_cross_page_merge,
        max_page_retries=max_page_retries,
        url=vllm_url,
        port=vllm_port,
    )
    
    logger.info(f"Configured to use vLLM server at {vllm_url}:{vllm_port}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Skip cross-page merge: {skip_cross_page_merge}")
    logger.info(f"Max page retries: {max_page_retries}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="OCRFlux Document Conversion API (Remote vLLM)",
    description="API server for converting PDFs and images to Markdown/text using remote vLLM server",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ConversionRequest(BaseModel):
    """Request model for URL-based conversion"""
    url: str = Field(..., description="URL of the PDF or image to convert")
    output_format: str = Field("markdown", description="Output format: 'markdown' or 'text'")
    max_page_retries: int = Field(1, description="Maximum retries for failed pages")
    skip_cross_page_merge: bool = Field(False, description="Skip cross-page merging")

class ConversionResponse(BaseModel):
    """Response model for conversion results"""
    status: str = Field(..., description="Conversion status: 'success' or 'error'")
    format: str = Field(..., description="Output format: 'markdown' or 'text'")
    content: Optional[str] = Field(None, description="Converted content")
    error: Optional[str] = Field(None, description="Error message if conversion failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global vllm_config
    
    # Try to check vLLM server health
    vllm_healthy = False
    if vllm_config:
        try:
            import requests
            response = requests.get(f"{vllm_config.url}:{vllm_config.port}/health", timeout=5)
            vllm_healthy = response.status_code == 200
        except:
            pass
    
    return {
        "status": "healthy",
        "vllm_configured": vllm_config is not None,
        "vllm_healthy": vllm_healthy,
        "timestamp": datetime.now().isoformat()
    }

def convert_markdown_to_text(markdown_text: str) -> str:
    """Convert markdown to plain text"""
    import re
    plain_text = markdown_text
    # Remove headers
    plain_text = re.sub(r'^#{1,6}\s+', '', plain_text, flags=re.MULTILINE)
    # Remove bold/italic
    plain_text = re.sub(r'\*{1,2}([^\*]+)\*{1,2}', r'\1', plain_text)
    # Remove links
    plain_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', plain_text)
    # Remove code blocks
    plain_text = re.sub(r'```[^`]*```', '', plain_text, flags=re.DOTALL)
    plain_text = re.sub(r'`([^`]+)`', r'\1', plain_text)
    # Remove images
    plain_text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', plain_text)
    # Remove horizontal rules
    plain_text = re.sub(r'^[-*_]{3,}$', '', plain_text, flags=re.MULTILINE)
    # Clean up extra whitespace
    plain_text = re.sub(r'\n{3,}', '\n\n', plain_text)
    return plain_text.strip()

@app.post("/convert/file", response_model=ConversionResponse)
async def convert_file(
    file: UploadFile = File(..., description="PDF or image file to convert"),
    output_format: str = Form("markdown", description="Output format: 'markdown' or 'text'"),
    max_page_retries: int = Form(1, description="Maximum retries for failed pages"),
    skip_cross_page_merge: bool = Form(False, description="Skip cross-page merging")
):
    """Convert uploaded PDF or image file to Markdown or text"""
    global vllm_config
    
    if not vllm_config:
        raise HTTPException(status_code=503, detail="vLLM server not configured")
    
    # Validate file type
    allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file to temp location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        logger.info(f"Processing file: {file.filename}")
        
        # Create custom config for this request
        request_config = Namespace(
            model=vllm_config.model,
            skip_cross_page_merge=skip_cross_page_merge,
            max_page_retries=max_page_retries,
            url=vllm_config.url,
            port=vllm_config.port,
        )
        
        # Call OCRFlux through vLLM server
        result = await ocrflux_request(request_config, tmp_file_path)
        
        if result is None:
            return ConversionResponse(
                status="error",
                format=output_format,
                error="Failed to parse document"
            )
        
        # Extract content
        document_text = result.get('document_text', '')
        
        # Convert to plain text if requested
        if output_format == "text":
            document_text = convert_markdown_to_text(document_text)
        
        # Prepare metadata
        metadata = {
            "filename": file.filename,
            "file_size": len(content),
            "pages_processed": result.get('num_pages', 0),
            "fallback_pages": result.get('fallback_pages', [])
        }
        
        return ConversionResponse(
            status="success",
            format=output_format,
            content=document_text,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return ConversionResponse(
            status="error",
            format=output_format,
            error=str(e)
        )
    finally:
        # Cleanup temp file
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@app.post("/convert/url", response_model=ConversionResponse)
async def convert_url(request: ConversionRequest):
    """Convert PDF or image from URL to Markdown or text"""
    global vllm_config
    
    if not vllm_config:
        raise HTTPException(status_code=503, detail="vLLM server not configured")
    
    try:
        import requests
        
        # Download file from URL
        response = requests.get(request.url, timeout=30)
        response.raise_for_status()
        
        # Determine file extension from URL or content-type
        from urllib.parse import urlparse
        url_path = urlparse(request.url).path
        file_ext = Path(url_path).suffix.lower()
        
        if not file_ext:
            # Try to guess from content-type
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' in content_type:
                file_ext = '.pdf'
            elif 'image' in content_type:
                file_ext = '.png'
            else:
                raise HTTPException(status_code=400, detail="Cannot determine file type from URL")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        
        logger.info(f"Processing URL: {request.url}")
        
        # Create custom config for this request
        request_config = Namespace(
            model=vllm_config.model,
            skip_cross_page_merge=request.skip_cross_page_merge,
            max_page_retries=request.max_page_retries,
            url=vllm_config.url,
            port=vllm_config.port,
        )
        
        # Call OCRFlux through vLLM server
        result = await ocrflux_request(request_config, tmp_file_path)
        
        if result is None:
            return ConversionResponse(
                status="error",
                format=request.output_format,
                error="Failed to parse document"
            )
        
        # Extract and format content
        document_text = result.get('document_text', '')
        
        if request.output_format == "text":
            document_text = convert_markdown_to_text(document_text)
        
        metadata = {
            "url": request.url,
            "pages_processed": result.get('num_pages', 0),
            "fallback_pages": result.get('fallback_pages', [])
        }
        
        return ConversionResponse(
            status="success",
            format=request.output_format,
            content=document_text,
            metadata=metadata
        )
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading URL: {str(e)}")
        return ConversionResponse(
            status="error",
            format=request.output_format,
            error=f"Failed to download URL: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        return ConversionResponse(
            status="error",
            format=request.output_format,
            error=str(e)
        )
    finally:
        # Cleanup temp file
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@app.get("/models")
async def list_models():
    """List available models from vLLM server"""
    global vllm_config
    
    if not vllm_config:
        raise HTTPException(status_code=503, detail="vLLM server not configured")
    
    try:
        import requests
        response = requests.get(f"{vllm_config.url}:{vllm_config.port}/v1/models", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Failed to fetch models: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8001"))
    
    uvicorn.run(app, host=host, port=port)