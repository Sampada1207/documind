"""OpenAI provider service for RAG response synthesis."""

import logging
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class OpenAIConfigurationError(Exception):
    """Raised when the OpenAI API key or required configuration is missing."""
    pass


class OpenAIAuthenticationError(Exception):
    """Raised when the OpenAI API rejects authentication (invalid API key)."""
    pass


class OpenAIRateLimitError(Exception):
    """Raised when the OpenAI request exceeds rate limits or quota."""
    pass


class OpenAINetworkError(Exception):
    """Raised when network timeouts or connection failures prevent reaching OpenAI."""
    pass


class OpenAIProviderError(Exception):
    """Raised when OpenAI returns an unexpected server-side error."""
    pass


class OpenAIService:
    """Manages OpenAI client lifecycle, authentication, and response generation."""

    def __init__(self) -> None:
        self._client = None
        self._mock_client = None

    def set_mock_client(self, mock_client) -> None:
        """Inject a mock client for testing without network requests or API costs."""
        self._mock_client = mock_client

    def _get_client(self):
        """Retrieve configured OpenAI client instance."""
        if self._mock_client is not None:
            return self._mock_client

        api_key = settings.OPENAI_API_KEY
        if not api_key or not api_key.strip():
            raise OpenAIConfigurationError(
                "OpenAI API key is not configured. Please set OPENAI_API_KEY in the backend environment (.env)."
            )

        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=api_key.strip(),
                    timeout=settings.OPENAI_TIMEOUT_SECONDS,
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {type(e).__name__}")
                raise OpenAIProviderError("Failed to initialize OpenAI client.") from e

        return self._client

    def generate_rag_answer(
        self,
        instructions: str,
        user_input: str,
        model: Optional[str] = None,
    ) -> str:
        """Generate answer from prompt and context using OpenAI Responses API.
        
        Args:
            instructions: System instructions and guardrails for grounded question answering.
            user_input: User question and retrieved context passages.
            model: Optional model override; defaults to settings.OPENAI_MODEL.
            
        Returns:
            Generated text string.
        """
        client = self._get_client()
        target_model = model or settings.OPENAI_MODEL

        try:
            # 1. Use the official Responses API if available
            if hasattr(client, "responses") and hasattr(client.responses, "create"):
                response = client.responses.create(
                    model=target_model,
                    instructions=instructions,
                    input=user_input,
                )
                if hasattr(response, "output_text") and response.output_text:
                    return response.output_text.strip()
                # Fallback extraction from response object
                if hasattr(response, "output") and response.output:
                    return str(response.output).strip()

            # 2. Fallback to Chat Completions API if Responses API is not available on client
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                response = client.chat.completions.create(
                    model=target_model,
                    messages=[
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": user_input},
                    ],
                )
                return response.choices[0].message.content.strip()

            raise OpenAIProviderError("OpenAI client does not support responses or chat completions.")

        except OpenAIConfigurationError:
            raise
        except Exception as e:
            error_type = type(e).__name__
            raw_str = str(e).lower()
            norm_str = raw_str.replace("_", " ")

            if any(term in norm_str for term in ["auth", "api key", "apikey", "unauthorized", "forbidden"]) or "authentication" in error_type.lower():
                logger.error(f"OpenAI authentication failure: {error_type}")
                raise OpenAIAuthenticationError(
                    "Invalid or unauthorized OpenAI API key. Please verify your OPENAI_API_KEY configuration."
                ) from e

            if any(term in norm_str for term in ["rate limit", "ratelimit", "quota", "too many requests"]) or "ratelimit" in error_type.lower():
                logger.warning(f"OpenAI rate limit / quota exceeded: {error_type}")
                raise OpenAIRateLimitError(
                    "OpenAI rate limit or usage quota exceeded. Please try again shortly."
                ) from e

            if any(term in norm_str for term in ["timeout", "timed out", "connection", "connect", "network"]) or "timeout" in error_type.lower() or "connection" in error_type.lower():
                logger.error(f"OpenAI network connectivity issue: {error_type}")
                raise OpenAINetworkError(
                    "Connection to OpenAI timed out or failed. Please check network connectivity."
                ) from e

            logger.error(f"OpenAI provider error: {error_type}")
            raise OpenAIProviderError(
                "An unexpected error occurred while communicating with the OpenAI service."
            ) from e


# Global service instance
openai_service = OpenAIService()
