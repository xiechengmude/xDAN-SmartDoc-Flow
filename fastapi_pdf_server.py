"""
High-performance FastAPI PDF Parser Server
Leverages existing OCRFlux infrastructure with optimizations
"""
import asyncio
import os
import tempfile
import shutil
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
from concurrent.futures import ThreadPoolExecutor
import uvloop

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import aiofiles
from contextlib import asynccontextmanager

# Import existing OCRFlux components
from argparse import Namespace
from ocrflux.client import request as ocrflux_request

# Use uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Configuration
class ServerConfig:
    VLLM_URL = os.getenv("VLLM_URL", "http://localhost")
    VLLM_PORT = int(os.getenv("VLLM_PORT", 8001))
    MODEL_NAME = os.getenv("MODEL_NAME", "ChatDOC/OCRFlux-3B")
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", 10))
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", 20))
    TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/pdf_parser")
    ENABLE_CROSS_PAGE_MERGE = os.getenv("ENABLE_CROSS_PAGE_MERGE", "true").lower() == "true"
    MAX_PAGE_RETRIES = int(os.getenv("MAX_PAGE_RETRIES", 3))

# Global resources
thread_pool = ThreadPoolExecutor(max_workers=ServerConfig.MAX_WORKERS)
semaphore = asyncio.Semaphore(ServerConfig.MAX_CONCURRENT_REQUESTS)
request_queue = asyncio.Queue(maxsize=100)

# Response models
class ParseResponse(BaseModel):
    job_id: str
    status: str = "processing"
    created_at: datetime
    document_text: Optional[str] = None
    page_texts: Optional[Dict[str, str]] = None
    num_pages: Optional[int] = None
    fallback_pages: Optional[List[int]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[ParseResponse] = None

# In-memory job storage (use Redis in production)
jobs_store: Dict[str, ParseResponse] = {}

# Lifespan manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    os.makedirs(ServerConfig.TEMP_DIR, exist_ok=True)
    print(f"Server starting with config:")
    print(f"  VLLM URL: {ServerConfig.VLLM_URL}:{ServerConfig.VLLM_PORT}")
    print(f"  Model: {ServerConfig.MODEL_NAME}")
    print(f"  Max concurrent requests: {ServerConfig.MAX_CONCURRENT_REQUESTS}")
    
    # Start background worker
    asyncio.create_task(background_worker())
    
    yield
    
    # Shutdown
    thread_pool.shutdown(wait=True)
    shutil.rmtree(ServerConfig.TEMP_DIR, ignore_errors=True)

# Create FastAPI app
app = FastAPI(
    title="OCRFlux PDF Parser API",
    description="High-performance PDF to Markdown conversion service",
    version="1.0.0",
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

# Background worker for processing queue
async def background_worker():
    """Process PDF parsing tasks from the queue"""
    while True:
        try:
            job_id, file_path, args = await request_queue.get()
            await process_pdf_task(job_id, file_path, args)
        except Exception as e:
            print(f"Worker error: {e}")
        finally:
            request_queue.task_done()

# PDF processing logic
async def process_pdf_task(job_id: str, file_path: str, args: Namespace):
    """Process a single PDF parsing task"""
    start_time = datetime.now()
    
    try:
        async with semaphore:
            # Call the existing OCRFlux client
            result = await ocrflux_request(args, file_path)
            
            if result:
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Update job status
                jobs_store[job_id].status = "completed"
                jobs_store[job_id].document_text = result['document_text']
                jobs_store[job_id].page_texts = result['page_texts']
                jobs_store[job_id].num_pages = result['num_pages']
                jobs_store[job_id].fallback_pages = result['fallback_pages']
                jobs_store[job_id].processing_time = processing_time
            else:
                jobs_store[job_id].status = "failed"
                jobs_store[job_id].error = "PDF parsing failed"
    
    except Exception as e:
        jobs_store[job_id].status = "failed"
        jobs_store[job_id].error = str(e)
    
    finally:
        # Clean up temp file
        try:
            os.remove(file_path)
        except:
            pass

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    queue_size = request_queue.qsize()
    return {
        "status": "healthy",
        "queue_size": queue_size,
        "max_queue_size": request_queue.maxsize,
        "active_jobs": len([j for j in jobs_store.values() if j.status == "processing"])
    }

@app.post("/parse", response_model=ParseResponse)
async def parse_pdf_async(
    file: UploadFile = File(...),
    skip_cross_page_merge: bool = Query(False, description="Skip cross-page element merging"),
    max_page_retries: int = Query(ServerConfig.MAX_PAGE_RETRIES, description="Max retries per page"),
    priority: bool = Query(False, description="High priority processing")
):
    """
    Upload a PDF file for asynchronous parsing
    Returns a job ID for status checking
    """
    # Validate file type
    if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only PDF and image files are supported")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    temp_path = os.path.join(ServerConfig.TEMP_DIR, f"{job_id}_{file.filename}")
    async with aiofiles.open(temp_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Create job entry
    job = ParseResponse(
        job_id=job_id,
        status="queued",
        created_at=datetime.now()
    )
    jobs_store[job_id] = job
    
    # Prepare arguments
    args = Namespace(
        model=ServerConfig.MODEL_NAME,
        url=ServerConfig.VLLM_URL,
        port=ServerConfig.VLLM_PORT,
        skip_cross_page_merge=skip_cross_page_merge or not ServerConfig.ENABLE_CROSS_PAGE_MERGE,
        max_page_retries=max_page_retries
    )
    
    # Add to processing queue
    try:
        if priority:
            # Priority jobs go to front of queue
            await request_queue.put((job_id, temp_path, args))
        else:
            await request_queue.put((job_id, temp_path, args))
        
        job.status = "processing"
    except asyncio.QueueFull:
        os.remove(temp_path)
        raise HTTPException(status_code=503, detail="Server is at capacity, please try again later")
    
    return job

@app.post("/parse-sync")
async def parse_pdf_sync(
    file: UploadFile = File(...),
    skip_cross_page_merge: bool = Query(False),
    max_page_retries: int = Query(ServerConfig.MAX_PAGE_RETRIES)
):
    """
    Synchronous PDF parsing endpoint
    Waits for processing to complete before returning
    """
    # Validate file
    if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only PDF and image files are supported")
    
    # Save file temporarily
    temp_path = os.path.join(ServerConfig.TEMP_DIR, f"sync_{uuid.uuid4()}_{file.filename}")
    async with aiofiles.open(temp_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    try:
        # Prepare arguments
        args = Namespace(
            model=ServerConfig.MODEL_NAME,
            url=ServerConfig.VLLM_URL,
            port=ServerConfig.VLLM_PORT,
            skip_cross_page_merge=skip_cross_page_merge or not ServerConfig.ENABLE_CROSS_PAGE_MERGE,
            max_page_retries=max_page_retries
        )
        
        # Process directly with semaphore control
        async with semaphore:
            result = await ocrflux_request(args, temp_path)
        
        if result:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=500, detail="PDF parsing failed")
    
    finally:
        # Clean up
        try:
            os.remove(temp_path)
        except:
            pass

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a parsing job"""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_store[job_id]
    
    # Calculate progress (simplified)
    progress = None
    if job.status == "processing":
        progress = 0.5  # In real implementation, track actual progress
    elif job.status == "completed":
        progress = 1.0
    
    return JobStatus(
        job_id=job_id,
        status=job.status,
        progress=progress,
        result=job if job.status in ["completed", "failed"] else None
    )

@app.get("/result/{job_id}/markdown")
async def get_markdown_result(job_id: str):
    """Get the parsed markdown text directly"""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_store[job_id]
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job is {job.status}")
    
    if not job.document_text:
        raise HTTPException(status_code=500, detail="No document text available")
    
    # Return as streaming response for large documents
    async def generate():
        yield job.document_text.encode('utf-8')
    
    return StreamingResponse(
        generate(),
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename=document_{job_id}.md"
        }
    )

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results"""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del jobs_store[job_id]
    return {"message": "Job deleted successfully"}

@app.get("/stats")
async def get_server_stats():
    """Get server statistics"""
    total_jobs = len(jobs_store)
    completed_jobs = len([j for j in jobs_store.values() if j.status == "completed"])
    failed_jobs = len([j for j in jobs_store.values() if j.status == "failed"])
    processing_jobs = len([j for j in jobs_store.values() if j.status == "processing"])
    
    avg_processing_time = 0
    if completed_jobs > 0:
        times = [j.processing_time for j in jobs_store.values() 
                if j.status == "completed" and j.processing_time]
        if times:
            avg_processing_time = sum(times) / len(times)
    
    return {
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "processing_jobs": processing_jobs,
        "queue_size": request_queue.qsize(),
        "avg_processing_time_seconds": avg_processing_time
    }

# Run with: uvicorn fastapi_pdf_server:app --host 0.0.0.0 --port 8000 --workers 1