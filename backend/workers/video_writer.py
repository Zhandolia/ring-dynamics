"""Browser-compatible H.264 output without OpenCV's optional H.264 encoder."""
import os
import shutil
import subprocess
import tempfile


class BrowserVideoWriter:
    def __init__(self, output_path, fps, size):
        binary = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
        if not binary:
            raise RuntimeError("FFmpeg is required to encode analysis videos. Install FFmpeg and retry.")
        width, height = size
        self._stderr = tempfile.TemporaryFile()
        self._closed = False
        self._process = subprocess.Popen([
            binary, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-threads", "1", "-movflags", "+faststart", output_path,
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr)

    def write(self, frame):
        try:
            self._process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            self.release()
            raise RuntimeError("FFmpeg stopped before the video was encoded.") from None

    def release(self):
        if self._closed:
            return
        self._closed = True
        try:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
            code = self._process.wait(timeout=120)
            if code:
                self._stderr.seek(0)
                message = self._stderr.read().decode(errors="replace")[-2000:]
                raise RuntimeError(f"FFmpeg could not encode the video: {message}")
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
            raise RuntimeError("Video encoding timed out.") from None
        finally:
            self._stderr.close()
