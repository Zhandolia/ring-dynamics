from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from uuid import uuid4
from datetime import datetime
import logging
import os
import json

from app.models.schemas import FightCreate, FightResponse
from app.core.config import settings
from app.services.annotation_service import run_annotation_background
from app.services.video_ingestion import normalize_youtube_url

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory storage (replace with database in production)
FIGHT_STORAGE = {}


@router.post("/fights/upload", response_model=FightResponse)
async def upload_fight_video(file: UploadFile = File(...)):
    """Upload a fight video for annotation analysis"""

    # Validate file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > settings.MAX_VIDEO_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.MAX_VIDEO_SIZE_MB}MB"
        )

    if file_size == 0:
        raise HTTPException(status_code=400, detail="The video file is empty.")

    # Validate file type
    allowed_types = ["video/mp4", "video/avi", "video/mov", "video/mkv",
                     "video/quicktime", "video/x-msvideo", "video/x-matroska", "video/webm"]
    generic_type = file.content_type in (None, "", "application/octet-stream")
    valid_extension = os.path.splitext(file.filename or "")[1].lower() in (
        ".mp4", ".avi", ".mov", ".mkv", ".webm"
    )
    if file.content_type not in allowed_types and not (generic_type and valid_extension):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Allowed: MP4, AVI, MOV, MKV"
        )

    # Generate fight ID
    fight_id = str(uuid4())

    # Save video file
    os.makedirs(settings.VIDEO_STORAGE_PATH, exist_ok=True)
    os.makedirs(settings.ANNOTATED_VIDEO_PATH, exist_ok=True)

    input_path = os.path.join(settings.VIDEO_STORAGE_PATH, f"{fight_id}.mp4")
    output_path = os.path.join(settings.ANNOTATED_VIDEO_PATH, f"{fight_id}_annotated.mp4")

    with open(input_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    logger.info(f"Video uploaded: {fight_id} ({file_size / 1024 / 1024:.1f} MB)")

    # Store initial status
    FIGHT_STORAGE[fight_id] = {
        "id": fight_id,
        "status": "pending",
        "created_at": datetime.now(),
        "video_url": input_path,
        "original_filename": file.filename,
    }

    # Start annotation in background
    run_annotation_background(
        fight_id=fight_id,
        input_path=input_path,
        output_path=output_path,
        fight_storage=FIGHT_STORAGE,
        device=settings.DEVICE,
        model_path=settings.YOLO_MODEL,
        scale=settings.ANNOTATION_SCALE,
        target_fps=settings.FRAMES_PER_SECOND,
        conf=settings.ANNOTATION_CONF,
        imgsz=settings.ANNOTATION_IMGSZ,
    )

    return FightResponse(
        id=fight_id,
        status="pending",
        created_at=FIGHT_STORAGE[fight_id]["created_at"],
        video_url=input_path,
    )


@router.post("/fights/youtube", response_model=FightResponse)
async def submit_youtube_video(request: FightCreate):
    """Queue one YouTube video through the upload annotation pipeline."""
    try:
        url = normalize_youtube_url(request.youtube_url or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    fight_id = str(uuid4())
    os.makedirs(settings.VIDEO_STORAGE_PATH, exist_ok=True)
    os.makedirs(settings.ANNOTATED_VIDEO_PATH, exist_ok=True)
    input_path = os.path.join(settings.VIDEO_STORAGE_PATH, f"{fight_id}.mp4")
    output_path = os.path.join(settings.ANNOTATED_VIDEO_PATH, f"{fight_id}_annotated.mp4")
    FIGHT_STORAGE[fight_id] = {
        "id": fight_id, "status": "pending", "created_at": datetime.now(),
        "video_url": input_path,
    }
    run_annotation_background(
        fight_id=fight_id, input_path=input_path, output_path=output_path,
        fight_storage=FIGHT_STORAGE, youtube_url=url,
        device=settings.DEVICE, model_path=settings.YOLO_MODEL,
        scale=settings.ANNOTATION_SCALE, target_fps=settings.FRAMES_PER_SECOND,
        conf=settings.ANNOTATION_CONF, imgsz=settings.ANNOTATION_IMGSZ,
    )
    return FightResponse(**FIGHT_STORAGE[fight_id])


@router.get("/fights/{fight_id}", response_model=FightResponse)
async def get_fight(fight_id: str):
    """Get fight analysis status and results"""
    if fight_id not in FIGHT_STORAGE:
        raise HTTPException(status_code=404, detail=f"Fight {fight_id} not found")

    data = FIGHT_STORAGE[fight_id]
    return FightResponse(
        id=data["id"],
        status=data["status"],
        created_at=data["created_at"],
        video_url=data.get("video_url"),
        annotated_video_url=data.get("annotated_video_url"),
        processing_time=data.get("processing_time"),
        metrics_url=data.get("metrics_url"),
        error=data.get("error"),
        total_rounds=data.get("total_rounds"),
        duration_seconds=data.get("duration_seconds"),
        win_probabilities=data.get("win_probabilities"),
        round_scores=data.get("round_scores"),
    )


@router.get("/fights/{fight_id}/progress")
async def get_fight_progress(fight_id: str):
    """Get detailed processing progress for pipeline UI"""
    if fight_id not in FIGHT_STORAGE:
        raise HTTPException(status_code=404, detail="Fight not found")

    data = FIGHT_STORAGE[fight_id]
    progress = data.get("progress", {
        "stage": 0, "stage_name": "Queued", "pct": 0,
        "frames_done": 0, "frames_total": 0,
    })
    return {
        "status": data["status"],
        "progress": progress,
    }


@router.get("/fights/{fight_id}/video")
async def get_annotated_video(fight_id: str):
    """Serve the annotated video file"""
    if fight_id not in FIGHT_STORAGE:
        raise HTTPException(status_code=404, detail="Fight not found")

    data = FIGHT_STORAGE[fight_id]
    if data["status"] != "completed":
        raise HTTPException(
            status_code=202,
            detail=f"Video not ready yet. Status: {data['status']}"
        )

    output_path = os.path.join(
        settings.ANNOTATED_VIDEO_PATH, f"{fight_id}_annotated.mp4"
    )
    if not os.path.isfile(output_path):
        raise HTTPException(status_code=404, detail="Annotated video file not found")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"{fight_id}_annotated.mp4",
    )


@router.get("/fights/{fight_id}/metrics")
async def get_fight_metrics(fight_id: str):
    """Serve the fight metrics JSON"""
    if fight_id not in FIGHT_STORAGE:
        raise HTTPException(status_code=404, detail="Fight not found")

    data = FIGHT_STORAGE[fight_id]
    if data["status"] != "completed":
        raise HTTPException(
            status_code=202,
            detail=f"Metrics not ready yet. Status: {data['status']}"
        )

    metrics_path = os.path.join(
        settings.ANNOTATED_VIDEO_PATH,
        f"{fight_id}_annotated_metrics.json"
    )
    if not os.path.isfile(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics file not found")

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    return JSONResponse(content=metrics)
