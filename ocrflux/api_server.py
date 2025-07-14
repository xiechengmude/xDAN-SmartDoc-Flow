import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ocrflux.adapters import (
    convert_api_request_to_ocrflux,
    convert_ocrflux_response_to_api,
    process_image_with_ocrflux
)
from ocrflux.config import get_config

app = FastAPI(
    title="OCRFlux API Server",
    description="API server for OCRFlux image recognition",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str | List[Dict[str, Any]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 4000
    temperature: Optional[float] = 0.1

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Process chat completion requests compatible with OpenAI API format
    """
    try:
        config = get_config()
        
        # Extract system prompt and image from request
        system_prompt = None
        image_data = None
        
        for message in request.messages:
            if message.role == "system":
                system_prompt = message.content
            elif message.role == "user":
                if isinstance(message.content, list):
                    for content_item in message.content:
                        if content_item.get("type") == "image_url":
                            image_url = content_item["image_url"]["url"]
                            if image_url.startswith("data:image"):
                                image_data = image_url.split(",")[1]
                        elif content_item.get("type") == "text":
                            # User text is handled separately if needed
                            pass
        
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Process image with OCRFlux
        result = await process_image_with_ocrflux(
            image_data=image_data,
            system_prompt=system_prompt,
            config=config
        )
        
        # Convert response to OpenAI format
        response = convert_ocrflux_response_to_api(
            result=result,
            model=request.model
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    config = get_config()
    uvicorn.run(
        "ocrflux.api_server:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.debug
    )