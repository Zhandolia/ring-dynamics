from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from uuid import uuid4
import logging
import os

from app.models.schemas import FightCreate, FightResponse
from app.core.config import settings
from app.core.websocket_manager import manager
from app.services.video_ingestion import process_video_upload, process_youtube_url

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory storage for demo (replace with database in production)
FIGHT_STORAGE = {}


@router.post("/fights/upload", response_model=FightResponse)
async def upload_fight_video(file: UploadFile = File(...)):
    """Upload a fight video for analysis"""
    
    # Validate file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset
    
    if file_size > settings.MAX_VIDEO_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.MAX_VIDEO_SIZE_MB}MB"
        )
    
    # Validate file type
    allowed_types = ["video/mp4", "video/avi", "video/mov", "video/mkv"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Generate fight ID
    fight_id = uuid4()
    
    # Save video file
    video_path = os.path.join(settings.VIDEO_STORAGE_PATH, f"{fight_id}.mp4")
    os.makedirs(settings.VIDEO_STORAGE_PATH, exist_ok=True)
    
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    logger.info(f"Video uploaded: {fight_id}")
    
    # Queue processing task
    # TODO: process_video_upload.delay(str(fight_id), video_path)
    
    return FightResponse(
        id=fight_id,
        status="pending",
        created_at=datetime.now(),
        video_url=video_path
    )


@router.post("/fights/youtube", response_model=FightResponse)
async def process_youtube_fight(fight: FightCreate):
    """Process a YouTube video URL"""
    
    if not fight.youtube_url:
        raise HTTPException(status_code=400, detail="YouTube URL required")
    
    fight_id = uuid4()
    
    logger.info(f"YouTube URL submitted: {fight_id} - {fight.youtube_url}")
    
    # Start download in background
    # In production, this would be a Celery task
    # For now, we'll call it directly (blocks API but works for demo)
    import threading
    import time
    def download_task():
        try:
            # Store initial status
            FIGHT_STORAGE[str(fight_id)] = {
                'id': fight_id,
                'status': 'processing',
                'created_at': datetime.now(),
                'video_url': fight.youtube_url
            }
            
            # Simulate download/processing
            time.sleep(5)  # Simulated processing time
            
            # Complete with mock results
            import random
            random.seed(42)
            FIGHT_STORAGE[str(fight_id)] = {
                'id': fight_id,
                'status': 'completed_mock',
                'created_at': datetime.now(),
                'video_url': fight.youtube_url,
                'total_rounds': 12,
                'duration_seconds': 2160,
                'win_probabilities': {
                    'fighter_0': random.uniform(0.4, 0.6),
                    'fighter_1': random.uniform(0.4, 0.6),
                    'draw': random.uniform(0.0, 0.1)
                },
                'round_scores': [
                    {
                        'round_number': i,
                        'probabilities': {
                            'prob_10_9_f0': random.uniform(0.3, 0.6),
                            'prob_10_9_f1': random.uniform(0.3, 0.6),
                            'prob_10_8_f0': random.uniform(0.0, 0.05),
                            'prob_10_8_f1': random.uniform(0.0, 0.05),
                            'prob_draw': random.uniform(0.05, 0.15)
                        }
                    } for i in range(1, 4)
                ]
            }
            logger.info(f"Mock processing complete for {fight_id}")
        except Exception as e:
            logger.error(f"Background download failed: {e}")
    
    thread = threading.Thread(target=download_task)
    thread.start()
    
    return FightResponse(
        id=fight_id,
        status="pending",
        created_at=datetime.now(),
        video_url=fight.youtube_url
    )


@router.get("/fights/{fight_id}", response_model=FightResponse)
async def get_fight(fight_id: str):
    """Get fight analysis status and results"""

    if fight_id in FIGHT_STORAGE:
        data = FIGHT_STORAGE[fight_id]
        return FightResponse(
            id=data['id'],
            status=data['status'],
            created_at=data['created_at'],
            video_url=data.get('video_url'),
            total_rounds=data.get('total_rounds'),
            duration_seconds=data.get('duration_seconds'),
            win_probabilities=data.get('win_probabilities'),
            round_scores=data.get('round_scores'),
        )

    # Fight not found
    raise HTTPException(status_code=404, detail=f"Fight {fight_id} not found")


@router.websocket("/ws/fights/{fight_id}")
async def websocket_endpoint(websocket: WebSocket, fight_id: str):
    """WebSocket endpoint for real-time fight updates"""
    
    await manager.connect(websocket, fight_id)
    
    try:
        # Send initial connection message
        await manager.send_personal_message({
            "type": "connected",
            "fight_id": fight_id,
            "message": "Connected to fight stream"
        }, websocket)
        
        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            logger.debug(f"Received from client: {data}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, fight_id)
        logger.info(f"Client disconnected from fight {fight_id}")


from datetime import datetime
