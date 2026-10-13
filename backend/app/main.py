from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import logging
import os

from app.api import fights
from app.core.config import settings
from app.core.video_stream import send_bytes_range_requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Ring Dynamics API...")
    # Ensure storage directories exist
    os.makedirs(settings.VIDEO_STORAGE_PATH, exist_ok=True)
    os.makedirs(settings.ANNOTATED_VIDEO_PATH, exist_ok=True)
    yield
    logger.info("Shutting down Ring Dynamics API...")


app = FastAPI(
    title="Ring Dynamics API",
    description="Boxing analytics platform with computer vision and Bayesian scoring",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Range", "Accept-Ranges", "Content-Length"],
)

def _safe_storage_path(base_dir: str, filename: str) -> str:
    """Resolve filename inside base_dir, rejecting directory traversal."""
    base = os.path.realpath(base_dir)
    file_path = os.path.realpath(os.path.join(base, filename))
    if os.path.commonpath([base, file_path]) != base:
        raise HTTPException(status_code=404, detail="File not found")
    return file_path


# Direct byte-range endpoints for video streaming / timeline seeking
@app.get("/storage/annotated/{filename:path}")
async def serve_annotated_video(filename: str, request: Request):
    file_path = _safe_storage_path(settings.ANNOTATED_VIDEO_PATH, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    if filename.lower().endswith((".mp4", ".mov", ".webm")):
        return send_bytes_range_requests(file_path, request.headers.get("range"))
    return FileResponse(file_path)


@app.get("/storage/videos/{filename:path}")
async def serve_raw_video(filename: str, request: Request):
    file_path = _safe_storage_path(settings.VIDEO_STORAGE_PATH, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    if filename.lower().endswith((".mp4", ".mov", ".webm")):
        return send_bytes_range_requests(file_path, request.headers.get("range"))
    return FileResponse(file_path)


# Include routers
app.include_router(fights.router, prefix="/api", tags=["fights"])

# Mount fallback storage
app.mount("/storage", StaticFiles(directory="storage", check_dir=False), name="storage")



@app.get("/")
async def root():
    return {"message": "Ring Dynamics API", "version": "2.0.0", "status": "operational"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
