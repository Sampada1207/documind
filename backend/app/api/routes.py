from fastapi import APIRouter
from app.core.config import settings
from app.api.documents import router as documents_router

router = APIRouter(prefix="/api/v1")

# Mount document processing routes
router.include_router(documents_router)


@router.get("/status", tags=["system"])
def api_status():
    """Get system status and metadata."""
    return {
        "status": "online",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "description": settings.DESCRIPTION,
    }

