import yt_dlp
import os
import logging

logger = logging.getLogger(__name__)


def download_youtube_video(youtube_url: str, output_path: str) -> str:
    """
    Download YouTube video in best quality
    
    Args:
        youtube_url: YouTube video URL
        output_path: Path to save video
        
    Returns:
        Path to downloaded video file
    """
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
        'quiet': False,
        'no_warnings': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Downloading YouTube video: {youtube_url}")
            ydl.download([youtube_url])
            logger.info(f"Download complete: {output_path}")
            return output_path
    except Exception as e:
        logger.error(f"YouTube download failed: {e}")
        raise


def process_video_upload(fight_id: str, video_path: str):
    """
    Process uploaded video file
    
    This will be converted to a Celery task
    """
    logger.info(f"Processing uploaded video: {fight_id}")
    # TODO: Queue to worker
    pass


def process_youtube_url(fight_id: str, youtube_url: str):
    """
    Download and process YouTube video
    
    This will be converted to a Celery task
    """
    logger.info(f"Processing YouTube URL for fight {fight_id}: {youtube_url}")
    
    # Create output path
    from app.core.config import settings
    os.makedirs(settings.VIDEO_STORAGE_PATH, exist_ok=True)
    output_file = os.path.join(settings.VIDEO_STORAGE_PATH, f"{fight_id}.mp4")
    
    try:
        # Download video
        downloaded_path = download_youtube_video(youtube_url, output_file)
        logger.info(f"Successfully downloaded video to: {downloaded_path}")
        
        # Trigger CV pipeline processing
        try:
            from workers.processor import FightProcessor
            processor = FightProcessor(device=settings.DEVICE)
            results = processor.process_fight(fight_id, downloaded_path)
            logger.info(f"Processing complete for {fight_id}: {results.get('status')}")
            return results
        except Exception as proc_error:
            logger.warning(f"CV processing failed, but video downloaded: {proc_error}")
            return {'status': 'downloaded', 'path': downloaded_path}
        
    except Exception as e:
        logger.error(f"Failed to process YouTube URL for fight {fight_id}: {e}")
        raise

