"""
HTTP 206 Partial Content video streaming helper for seekable HTML5 video playback.
"""
import os
from typing import Optional
from fastapi import HTTPException
from fastapi.responses import StreamingResponse, Response


def send_bytes_range_requests(file_path: str, range_header: Optional[str] = None) -> Response:
    """
    Serve a video file with RFC 7233 byte-range support.
    Enables seeking, scrubbing, and fast loading in all HTML5 video players.
    """
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    file_size = os.path.getsize(file_path)

    if not range_header or not range_header.strip().lower().startswith("bytes="):
        def full_iter():
            with open(file_path, "rb") as f:
                while chunk := f.read(256 * 1024):
                    yield chunk

        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": "video/mp4",
        }
        return StreamingResponse(full_iter(), status_code=200, headers=headers)

    # Parse range header: "bytes=start-end"
    range_val = range_header.strip()[6:]  # remove 'bytes='
    parts = range_val.split("-")

    start_str = parts[0].strip() if len(parts) > 0 else ""
    end_str = parts[1].strip() if len(parts) > 1 else ""

    try:
        if start_str and end_str:
            start = int(start_str)
            end = int(end_str)
        elif start_str:
            start = int(start_str)
            end = file_size - 1
        elif end_str:
            end = file_size - 1
            start = max(0, file_size - int(end_str))
        else:
            start = 0
            end = file_size - 1
    except ValueError:
        raise HTTPException(
            status_code=416,
            detail="Invalid range header values",
            headers={"Content-Range": f"bytes */{file_size}"}
        )

    if start >= file_size or end >= file_size or start > end:
        raise HTTPException(
            status_code=416,
            detail="Requested range not satisfiable",
            headers={"Content-Range": f"bytes */{file_size}"}
        )

    content_length = end - start + 1

    def range_iter():
        with open(file_path, "rb") as f:
            f.seek(start)
            bytes_left = content_length
            chunk_size = 256 * 1024
            while bytes_left > 0:
                to_read = min(chunk_size, bytes_left)
                data = f.read(to_read)
                if not data:
                    break
                bytes_left -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": "video/mp4",
    }
    return StreamingResponse(range_iter(), status_code=206, headers=headers)
