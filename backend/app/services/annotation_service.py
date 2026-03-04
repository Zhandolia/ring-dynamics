"""
Annotation Service — runs annotate_video.py on uploaded fight clips.

Wraps the annotation pipeline as a background task that updates
FIGHT_STORAGE with status and the annotated video URL.
"""

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


def run_annotation_background(
    fight_id: str,
    input_path: str,
    output_path: str,
    fight_storage: dict,
    device: str = "mps",
    model_path: str = "models/yolov8n.pt",
    scale: float = 0.5,
    target_fps: float = 30,
    conf: float = 0.30,
    imgsz: int = 640,
):
    """
    Run annotation in a background thread.

    Updates fight_storage[fight_id] with:
      - status: 'annotating' → 'completed' (or 'failed')
      - annotated_video_url: relative path to annotated MP4
      - processing_time: seconds elapsed
    """

    def _worker():
        t0 = time.time()
        try:
            # Update status
            if fight_id in fight_storage:
                fight_storage[fight_id]["status"] = "annotating"
                fight_storage[fight_id]["progress"] = {
                    "stage": 0,
                    "stage_name": "Queued",
                    "pct": 0,
                    "frames_done": 0,
                    "frames_total": 0,
                }

            logger.info(f"[Annotation] Starting: {fight_id}")
            logger.info(f"  Input:  {input_path}")
            logger.info(f"  Output: {output_path}")
            logger.info(f"  Model:  {model_path}  Device: {device}")

            # Progress callback
            def on_progress(stage: int, stage_name: str, pct: float,
                            frames_done: int = 0, frames_total: int = 0):
                if fight_id in fight_storage:
                    fight_storage[fight_id]["progress"] = {
                        "stage": stage,
                        "stage_name": stage_name,
                        "pct": round(pct, 1),
                        "frames_done": frames_done,
                        "frames_total": frames_total,
                    }

            # Import the annotation function
            import sys
            backend_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)
            )))
            if backend_root not in sys.path:
                sys.path.insert(0, backend_root)

            from workers.annotate_video import annotate_video

            # Run annotation
            annotate_video(
                input_path=input_path,
                output_path=output_path,
                model_name=model_path,
                conf=conf,
                device=device,
                imgsz=imgsz,
                scale=scale,
                target_fps=target_fps,
                progress_cb=on_progress,
            )

            elapsed = time.time() - t0
            logger.info(f"[Annotation] Completed: {fight_id} in {elapsed:.1f}s")

            # Update storage
            if fight_id in fight_storage:
                fight_storage[fight_id]["status"] = "completed"
                fight_storage[fight_id]["processing_time"] = round(elapsed, 1)
                filename = os.path.basename(output_path)
                fight_storage[fight_id]["annotated_video_url"] = (
                    f"/storage/annotated/{filename}"
                )
                metrics_filename = filename.rsplit('.', 1)[0] + '_metrics.json'
                fight_storage[fight_id]["metrics_url"] = (
                    f"/storage/annotated/{metrics_filename}"
                )

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"[Annotation] Failed: {fight_id} after {elapsed:.1f}s — {e}")
            if fight_id in fight_storage:
                fight_storage[fight_id]["status"] = "failed"
                fight_storage[fight_id]["error"] = str(e)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread
