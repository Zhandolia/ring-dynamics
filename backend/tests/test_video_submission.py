"""Regression coverage for real submission failures, without network downloads."""
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api import fights
from app.core.config import settings
from app.main import app
from app.services.video_ingestion import normalize_youtube_url, download_youtube_video
from app.services import annotation_service, video_ingestion


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "VIDEO_STORAGE_PATH", str(tmp_path / "storage/videos"))
    monkeypatch.setattr(settings, "ANNOTATED_VIDEO_PATH", str(tmp_path / "storage/annotated"))
    fights.FIGHT_STORAGE.clear()
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("url", [
    "https://youtube.com/watch?v=BaW_jenozKc&list=ignored",
    "https://youtu.be/BaW_jenozKc?t=10",
    "https://www.youtube.com/shorts/BaW_jenozKc",
    "https://m.youtube.com/watch?v=BaW_jenozKc",
])
def test_youtube_urls_are_canonical_single_videos(url):
    assert normalize_youtube_url(url) == "https://www.youtube.com/watch?v=BaW_jenozKc"


@pytest.mark.parametrize("url", [
    "", "https://youtube.com/playlist?list=abc", "https://youtube.com.evil.test/watch?v=BaW_jenozKc",
    "http://127.0.0.1/video.mp4", "file:///private", "https://youtube.com/watch?v=bad",
    "https://user:secret@youtube.com/watch?v=BaW_jenozKc",
])
def test_invalid_youtube_is_rejected_before_queueing(client, monkeypatch, url):
    monkeypatch.setattr(fights, "run_annotation_background", lambda **kw: pytest.fail("queued invalid URL"))
    response = client.post("/api/fights/youtube", json={"youtube_url": url})
    assert response.status_code == 400
    assert not fights.FIGHT_STORAGE


def test_youtube_route_queues_annotation_and_returns_pollable_id(client, monkeypatch):
    submitted = []
    monkeypatch.setattr(fights, "run_annotation_background", lambda **kw: submitted.append(kw))
    response = client.post("/api/fights/youtube", json={"youtube_url": "https://youtu.be/BaW_jenozKc"})
    assert response.status_code == 200
    result = response.json()
    assert client.get(f"/api/fights/{result['id']}").json()["status"] == "pending"
    assert submitted[0]["youtube_url"] == "https://www.youtube.com/watch?v=BaW_jenozKc"
    assert submitted[0]["fight_storage"] is fights.FIGHT_STORAGE


@pytest.mark.parametrize("name,mime", [("fight.mp4", "video/mp4"), ("fight.mkv", "video/x-matroska"), ("fight.mov", "application/octet-stream")])
def test_upload_preserves_bytes_and_queues(client, monkeypatch, name, mime):
    submitted = []
    monkeypatch.setattr(fights, "run_annotation_background", lambda **kw: submitted.append(kw))
    payload = b"sample-upload-bytes"
    response = client.post("/api/fights/upload", files={"file": (name, payload, mime)})
    assert response.status_code == 200
    assert Path(submitted[0]["input_path"]).read_bytes() == payload


def test_empty_and_oversized_uploads_do_not_queue(client, monkeypatch):
    monkeypatch.setattr(fights, "run_annotation_background", lambda **kw: pytest.fail("queued invalid upload"))
    assert client.post("/api/fights/upload", files={"file": ("empty.mp4", b"", "video/mp4")}).status_code == 400
    monkeypatch.setattr(settings, "MAX_VIDEO_SIZE_MB", 0)
    assert client.post("/api/fights/upload", files={"file": ("large.mp4", b"x", "video/mp4")}).status_code == 413


def test_failed_analysis_exposes_error(client):
    fight_id = str(uuid4())
    fights.FIGHT_STORAGE[fight_id] = {"id": fight_id, "created_at": datetime.now(), "status": "failed", "error": "Cannot decode this video."}
    response = client.get(f"/api/fights/{fight_id}")
    assert response.json()["error"] == "Cannot decode this video."


def test_download_failure_is_terminal_and_visible(client, monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError("YouTube restricted this video. Upload the file instead.")
    monkeypatch.setattr(video_ingestion, "download_youtube_video", fail)
    fight_id = str(uuid4())
    fights.FIGHT_STORAGE[fight_id] = {"id": fight_id, "created_at": datetime.now(), "status": "pending"}
    future = annotation_service.run_annotation_background(
        fight_id, str(tmp_path / "input.mp4"), str(tmp_path / "output.mp4"),
        fights.FIGHT_STORAGE, youtube_url="https://youtu.be/BaW_jenozKc",
    )
    future.result(timeout=5)
    result = client.get(f"/api/fights/{fight_id}").json()
    assert result["status"] == "failed"
    assert "Upload the file" in result["error"]
    assert not result["annotated_video_url"]


def test_downloader_cleans_partial_file_and_never_claims_success(monkeypatch, tmp_path):
    output = tmp_path / "video.mp4"
    class UnavailableDownloader:
        def __init__(self, options):
            assert options["noplaylist"] is True
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def extract_info(self, *args, **kwargs):
            Path(str(output) + ".part").write_bytes(b"partial")
            raise RuntimeError("Video unavailable")
    monkeypatch.setattr(video_ingestion.yt_dlp, "YoutubeDL", UnavailableDownloader)
    with pytest.raises(RuntimeError, match="Try uploading"):
        download_youtube_video("https://youtu.be/BaW_jenozKc", str(output), 1024)
    assert not Path(str(output) + ".part").exists()
