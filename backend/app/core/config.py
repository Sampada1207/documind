import os
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# Prevent macOS libomp conflict when torch and faiss are loaded together
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class Settings:
    """Application settings loaded from environment and .env."""

    def __init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        """Reload configuration from environment variables."""
        self.PROJECT_NAME: str = os.getenv("APP_NAME", "DocuMind API")
        self.VERSION: str = "0.1.0"
        self.DESCRIPTION: str = "Intelligent Document Research Assistant API"
        self.ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

        # Default allowed origins for frontend local dev server
        _raw_cors = os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
        )
        self.CORS_ORIGINS: List[str] = [
            origin.strip() for origin in _raw_cors.split(",") if origin.strip()
        ]

        # LLM Provider: 'mock' (default for free local testing) or 'openai'
        raw_provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
        if raw_provider in ("mock", "openai"):
            self.LLM_PROVIDER: str = raw_provider
        else:
            self.LLM_PROVIDER = "mock"

        # OpenAI Configuration
        raw_key = os.getenv("OPENAI_API_KEY")
        self.OPENAI_API_KEY: Optional[str] = raw_key.strip() if (raw_key and raw_key.strip()) else None
        self.OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

        # Timeout parsing with fallback
        try:
            self.OPENAI_TIMEOUT_SECONDS: float = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30.0"))
        except (ValueError, TypeError):
            self.OPENAI_TIMEOUT_SECONDS = 30.0

    @property
    def is_mock_mode(self) -> bool:
        """Convenience property to check if mock LLM provider is active."""
        return self.LLM_PROVIDER == "mock"


settings = Settings()
