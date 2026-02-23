from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import os

from app.api import fights
from app.core.config import settings

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
)

# Include routers
app.include_router(fights.router, prefix="/api", tags=["fights"])

# Mount storage for serving annotated videos
app.mount("/storage", StaticFiles(directory="storage"), name="storage")


@app.get("/")
async def root():
    return {"message": "Ring Dynamics API", "version": "2.0.0", "status": "operational"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
