import base64
import json
import time
import uuid
import asyncio
from io import BytesIO
from typing import Dict, Any, Optional
from PIL import Image
from argparse import Namespace

from ocrflux.client import request

async def process_image_with_ocrflux(
    image_data: str,
    system_prompt: Optional[str],
    config: Any
) -> Dict[str, Any]:
    """
    Process image with OCRFlux backend
    
    Args:
        image_data: Base64 encoded image data
        system_prompt: System prompt for guidance
        config: Configuration object
    
    Returns:
        OCRFlux processing result
    """
    # Decode base64 image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    
    # Save temporary image file (OCRFlux expects file path)
    temp_image_path = f"/tmp/ocrflux_temp_{uuid.uuid4()}.png"
    image.save(temp_image_path, format="PNG")
    
    try:
        # Prepare arguments for OCRFlux client
        args = Namespace(
            model=config.ocrflux_model,
            skip_cross_page_merge=True,  # Single image, no cross-page merge needed
            max_page_retries=config.max_retries,
            url=config.vllm_url,
            port=config.vllm_port,
        )
        
        # Call OCRFlux processing
        result = await request(args, temp_image_path)
        
        if result is None:
            raise Exception("OCRFlux processing failed")
        
        # Extract markdown text
        markdown_text = result.get("document_text", "")
        
        # If system prompt requests specific format, apply it
        if system_prompt and "列表" in system_prompt:
            # Already in markdown format, just ensure it's properly formatted
            formatted_text = format_as_list(markdown_text)
        else:
            formatted_text = markdown_text
        
        return {
            "content": formatted_text,
            "original_result": result
        }
        
    finally:
        # Clean up temporary file
        import os
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

def format_as_list(text: str) -> str:
    """
    Format text as markdown list, handling checkboxes
    """
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Handle checkboxes
        if "☑" in line or "✓" in line or "√" in line:
            line = line.replace("☑", "[√]").replace("✓", "[√]").replace("√", "[√]")
        elif "☐" in line or "□" in line:
            line = line.replace("☐", "[ ]").replace("□", "[ ]")
        
        # Ensure line starts with list marker if not already
        if not line.startswith(('-', '*', '•', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
            line = f"- {line}"
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def convert_ocrflux_response_to_api(
    result: Dict[str, Any],
    model: str
) -> Dict[str, Any]:
    """
    Convert OCRFlux response to OpenAI API format
    """
    content = result.get("content", "")
    
    # Create response in OpenAI format
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,  # Estimated
            "completion_tokens": len(content.split()),  # Rough estimate
            "total_tokens": 100 + len(content.split())
        }
    }
    
    return response

def convert_api_request_to_ocrflux(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert API request format to OCRFlux format
    Note: This function is kept for potential future use
    """
    # Extract relevant data from API request
    messages = request_data.get("messages", [])
    
    # Find system prompt and image
    system_prompt = None
    image_data = None
    
    for message in messages:
        if message["role"] == "system":
            system_prompt = message["content"]
        elif message["role"] == "user":
            if isinstance(message["content"], list):
                for content in message["content"]:
                    if content["type"] == "image_url":
                        image_data = content["image_url"]["url"]
    
    return {
        "system_prompt": system_prompt,
        "image_data": image_data,
        "temperature": request_data.get("temperature", 0.1),
        "max_tokens": request_data.get("max_tokens", 4000)
    }