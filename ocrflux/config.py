import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Configuration for OCRFlux API Server"""
    
    # API Server settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # vLLM Server settings
    vllm_url: str = "http://localhost"
    vllm_port: int = 8003
    
    # Model settings
    ocrflux_model: str = "ChatDOC/OCRFlux-3B"
    max_retries: int = 1
    
    # Response settings
    default_temperature: float = 0.1
    default_max_tokens: int = 4000
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables"""
        return cls(
            api_host=os.getenv("API_HOST", cls.api_host),
            api_port=int(os.getenv("API_PORT", cls.api_port)),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            vllm_url=os.getenv("VLLM_URL", cls.vllm_url),
            vllm_port=int(os.getenv("VLLM_PORT", cls.vllm_port)),
            ocrflux_model=os.getenv("OCRFLUX_MODEL", cls.ocrflux_model),
            max_retries=int(os.getenv("MAX_RETRIES", cls.max_retries)),
            default_temperature=float(os.getenv("DEFAULT_TEMPERATURE", cls.default_temperature)),
            default_max_tokens=int(os.getenv("DEFAULT_MAX_TOKENS", cls.default_max_tokens))
        )

# Global config instance
_config: Optional[Config] = None

def get_config() -> Config:
    """Get or create config instance"""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config

def set_config(config: Config):
    """Set global config instance"""
    global _config
    _config = config