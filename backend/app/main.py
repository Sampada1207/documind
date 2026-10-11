import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import router as api_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
)

# Safe CORS configuration: wildcard origin forbids allow_credentials per W3C specification
safe_credentials = "*" not in settings.CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=safe_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_unhandled_exception_handler(request: Request, exc: Exception):
    """Global handler for uncaught exceptions to prevent leaking stack traces or internal secrets."""
    logger.error(
        f"Unhandled server error on {request.method} {request.url.path}: {type(exc).__name__} - {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected internal server error occurred. Please contact the administrator."},
    )


# Modular API router
app.include_router(api_router)



@app.get("/", tags=["root"])
def read_root():
    """Root endpoint returning application information."""
    return {
        "message": "Welcome to DocuMind API - Intelligent Document Research Assistant",
        "status": "running",
        "version": settings.VERSION,
    }


@app.get("/health", tags=["health"])
def health_check():
    """Health check endpoint to verify backend operational status."""
    return {
        "status": "healthy",
        "service": "documind-backend",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "llm_provider": settings.LLM_PROVIDER,
    }
