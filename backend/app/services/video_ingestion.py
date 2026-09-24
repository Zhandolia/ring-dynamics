"""Download one public YouTube video for the annotation worker."""
import logging
import os
import re
from urllib.parse import urlsplit, parse_qs
import yt_dlp

logger = logging.getLogger(__name__)


def normalize_youtube_url(value: str) -> str:
    try:
        url = urlsplit(value.strip())
        host = (url.hostname or "").lower()
        if url.scheme not in ("http", "https") or url.username or url.password or url.port:
            raise ValueError
        parts = url.path.strip("/").split("/")
        if host in ("youtu.be", "www.youtu.be") and len(parts) == 1:
            video_id = parts[0]
        elif host in ("youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"):
            if url.path == "/watch":
                video_id = parse_qs(url.query).get("v", [""])[0]
            elif len(parts) == 2 and parts[0] in ("shorts", "embed", "live"):
                video_id = parts[1]
            else:
                raise ValueError
        else:
            raise ValueError
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError
        return f"https://www.youtube.com/watch?v={video_id}"
    except ValueError:
        raise ValueError("Enter a valid YouTube video link (not a channel or playlist).") from None


def download_youtube_video(youtube_url: str, output_path: str, max_bytes: int,
                           progress_cb=None) -> str:
    url = normalize_youtube_url(youtube_url)

    def progress(data):
        downloaded = data.get("downloaded_bytes", 0)
        total = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
        if downloaded > max_bytes:
            raise ValueError("YouTube video exceeds the maximum video size.")
        if progress_cb:
            pct = min(99, downloaded / total * 100) if total else 0
            progress_cb(0, f"Downloading YouTube video ({pct:.0f}%)", 0)

    def reject_live(info, *, incomplete=False):
        if info.get("is_live"):
            return "Live streams cannot be analyzed. Choose a completed video."

    options = {
        # Analysis is silent; a separate audio download is unnecessary.
        "format": "bestvideo[height<=720][ext=mp4]/best[height<=720][ext=mp4]/bestvideo[height<=720]/best[height<=720]",
        "outtmpl": output_path, "noplaylist": True, "max_filesize": max_bytes,
        "socket_timeout": 30, "retries": 2, "fragment_retries": 2,
        "quiet": True, "noprogress": True, "progress_hooks": [progress],
        "match_filter": reject_live, "js_runtimes": {"deno": {}, "node": {}},
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
        if not info or not os.path.isfile(output_path) or not os.path.getsize(output_path):
            raise ValueError("Video was unavailable, live, or larger than the upload limit.")
        if os.path.getsize(output_path) > max_bytes:
            raise ValueError("YouTube video exceeds the maximum video size.")
        return output_path
    except Exception as exc:
        logger.warning("YouTube download failed: %s", exc)
        for suffix in ("", ".part", ".ytdl"):
            path = output_path + suffix
            if os.path.isfile(path):
                os.remove(path)
        raise RuntimeError(
            "YouTube download failed. The video may be private, restricted, or blocked by "
            "YouTube on this server. Try uploading the video file instead."
        ) from exc
