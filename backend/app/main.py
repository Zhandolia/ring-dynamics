from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.api import fights
from app.core.config import settings
from app.core.websocket_manager import manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Ring Dynamics API...")
    yield
    logger.info("Shutting down Ring Dynamics API...")


app = FastAPI(
    title="Ring Dynamics API",
    description="Boxing analytics platform with computer vision and Bayesian scoring",
    version="1.0.0",
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


@app.get("/")
async def root():
    return {"message": "Ring Dynamics API", "status": "operational"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
