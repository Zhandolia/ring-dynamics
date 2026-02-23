# 🥊 Ring Dynamics

**AI-powered boxing analytics platform** — real-time fighter tracking, live scoring overlays, and fight metrics visualization.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![Next.js](https://img.shields.io/badge/next.js-14-black) ![FastAPI](https://img.shields.io/badge/fastapi-0.104-green) ![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-purple)

<br>

## What It Does

Upload a boxing video → Ring Dynamics runs YOLOv8 fighter detection with hardened 2-fighter tracking, draws shivering 3-box overlays (full body, head, core), computes real-time fight metrics, and renders a premium scoring HUD — all viewable in a split-layout web dashboard with live stats synced to video playback.

### Key Features

| Feature | Description |
|---------|-------------|
| **3-Box Fighter Tracking** | Full body + head + core sub-boxes with ±2px jitter shimmer for a "live tracking" feel |
| **Anti-Audience Filtering** | Spatial reasoning + size constraints ensure only the two fighters are tracked |
| **Identity Lock** | Appearance-based + spatial matching keeps Fighter A / Fighter B consistent across camera cuts |
| **Live Scoring Overlay** | Activity, Aggression, Ring Control, Pressure — rendered as side panels + top/bottom bars on the video |
| **10-9 Round Scoring** | Weighted composite metrics produce estimated round scores |
| **Event Feed** | Auto-generated fight events ("Fighter B presses forward", "Fighter A controls center") |
| **Metrics JSON Export** | Per-second timeline data saved alongside annotated videos |
| **Split-Layout Dashboard** | Video player (70%) + live stats panel (30%) with red/black premium theme |

<br>

## Architecture

```
┌────────────────────┐        ┌───────────────────┐        ┌──────────────────┐
│   Next.js 14 UI    │◄──────►│  FastAPI Backend   │───────►│  CV Pipeline     │
│   (React/TS)       │  REST  │   (Orchestrator)   │ Thread │  (annotate_video)│
│                    │        │                    │        │                  │
│  • Split layout    │        │  • Video upload    │        │  • YOLOv8        │
│  • Live stats      │        │  • Status polling  │        │  • ByteTrack     │
│  • Event feed      │        │  • Metrics API     │        │  • FightScorer   │
│  • Scorecard       │        │  • Video serving   │        │  • 3-box drawing │
└────────────────────┘        └───────────────────┘        └──────────────────┘
```

<br>

## Quick Start

### Prerequisites

- **Python 3.10+** with `pip`
- **Node.js 18+** with `npm`
- (Optional) **Apple MPS** or **NVIDIA GPU** for faster inference

### 1. Clone & Install

```bash
git clone https://github.com/Zhandolia/ring-dynamics.git
cd ring-dynamics

# Backend dependencies
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend dependencies
cd ../frontend
npm install
```

### 2. Start the Backend

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Visit `/docs` for the interactive Swagger UI.

### 3. Start the Frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000` in your browser.

### 4. Upload & Analyze

1. Click **"Upload Video"** on the home page
2. Select any boxing video (MP4, AVI, MOV — up to 500MB)
3. Watch the processing status animate
4. Once complete, the **split-layout results page** appears:
   - **Left (~70%)** — Annotated video with scoring HUD overlay
   - **Right (~30%)** — Live stats panel synced to video playback

<br>

## Standalone CLI

You can also run the annotation pipeline directly without the web UI:

```bash
# Basic usage
python3 annotate_video.py input.mp4 output.mp4

# With GPU acceleration and custom settings
python3 annotate_video.py fight.mp4 annotated.mp4 \
  --device mps \
  --scale 0.5 \
  --conf 0.30 \
  --imgsz 640 \
  --fps 30
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | — | Path to input video (required) |
| `output` | auto | Output path (defaults to `{input}_annotated.mp4`) |
| `--model` | `yolov8n.pt` | YOLOv8 model variant (`n`, `s`, `m`, `l`, `x`) |
| `--device` | `cpu` | Inference device (`cpu`, `mps`, `cuda`) |
| `--scale` | `1.0` | Frame resize scale (0.5 = half size, faster) |
| `--conf` | `0.35` | Detection confidence threshold |
| `--imgsz` | `1280` | YOLO input resolution |
| `--fps` | `0` | Target output FPS (0 = match source) |
| `--max-frames` | `0` | Limit frames to process (0 = all) |

The CLI also generates a `_metrics.json` file alongside the output video.

<br>

## API Reference

### Upload Video

```bash
curl -X POST \
  -F "file=@fight.mp4;type=video/mp4" \
  http://localhost:8000/api/fights/upload
```

**Response:**
```json
{
  "id": "8f6b9f0e-...",
  "status": "pending",
  "metrics_url": null
}
```

### Check Status

```bash
curl http://localhost:8000/api/fights/{fight_id}
```

Status transitions: `pending` → `annotating` → `completed` (or `failed`)

### Get Annotated Video

```bash
curl http://localhost:8000/api/fights/{fight_id}/video -o annotated.mp4
```

### Get Fight Metrics

```bash
curl http://localhost:8000/api/fights/{fight_id}/metrics
```

**Returns per-second timeline data:**
```json
{
  "duration": 180.0,
  "final_scores": [10, 9],
  "final_activity": [1.52, 1.81],
  "final_aggression": [0.99, 0.43],
  "events": [
    {"frame": 30, "text": "Fighter B presses forward"},
    {"frame": 150, "text": "Fighter A controls center"}
  ],
  "timeline": [
    {"time": 1.0, "activity": [35.2, 71.5], "distance": "Outside", ...}
  ]
}
```

<br>

## Project Structure

```
ring-dynamics/
├── annotate_video.py              # Main CV pipeline (standalone CLI)
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI application
│   │   ├── api/
│   │   │   └── fights.py          # REST endpoints (upload, status, video, metrics)
│   │   ├── core/
│   │   │   └── config.py          # Settings & environment config
│   │   ├── models/
│   │   │   └── schemas.py         # Pydantic data models
│   │   └── services/
│   │       └── annotation_service.py  # Background annotation runner
│   ├── workers/
│   │   └── annotate_video.py      # Backend copy of CV pipeline
│   └── requirements.txt
├── frontend/
│   ├── src/app/
│   │   ├── page.tsx               # Home page (upload UI)
│   │   ├── fight/[id]/page.tsx    # Fight results (split layout)
│   │   ├── globals.css            # Red/black theme
│   │   └── layout.tsx             # Root layout
│   ├── package.json
│   └── tsconfig.json
└── README.md
```

<br>

## How the CV Pipeline Works

### Detection & Tracking

1. **YOLOv8** detects all people in each frame with bounding boxes
2. **ByteTrack** assigns persistent track IDs across frames
3. **Anti-audience filter** selects the two largest, most central detections
4. **Identity tracker** uses spatial proximity + appearance histograms to maintain consistent Fighter A / Fighter B assignment even through camera cuts

### 3-Box Visualization

Each fighter gets three overlapping boxes drawn with ±2px jitter per frame:
- **Full body** (outer box) — fighter-colored (red/blue)
- **Head zone** (top 0-28%) — green accent
- **Core zone** (28-62%) — fighter-colored

The jitter creates a "shivering" effect that makes the tracking feel alive and millisecond-precise.

### Scoring Overlay (HUD)

The video canvas is expanded to include:
- **Top bar** — Fighter names, round scores (10-9 system), round number, timer
- **Side panels** — Per-fighter Activity, Aggression, Ring Control, Pressure bars
- **Bottom bar** — Distance gauge (Inside/Mid-Range/Outside) + latest event

### Metrics Computation

| Metric | How It's Calculated |
|--------|---------------------|
| **Activity** | Fighter bbox center movement speed (EMA smoothed) |
| **Aggression** | Forward movement toward the opponent |
| **Ring Control** | Proximity to ring center (closer = higher) |
| **Pressure** | Sustained forward movement over time |
| **Distance** | Gap between fighter bboxes (Inside/Mid/Outside) |
| **Round Score** | Weighted composite: 40% activity + 35% aggression + 25% ring control |

<br>

## Configuration

Environment variables (optional — sane defaults used):

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `mps` | Inference device (`cpu`, `mps`, `cuda`) |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model name |
| `ANNOTATION_SCALE` | `0.5` | Frame resize factor |
| `ANNOTATION_CONF` | `0.30` | Detection confidence |
| `ANNOTATION_IMGSZ` | `640` | YOLO input size |
| `MAX_VIDEO_SIZE_MB` | `500` | Max upload file size |
| `FRAMES_PER_SECOND` | `30` | Target output FPS |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL for frontend |

<br>

## Roadmap

- [ ] Fine-tune YOLOv8 on boxing-specific dataset (fighters, gloves, referee)
- [ ] Train punch detection model (jab, cross, hook, uppercut)
- [ ] Implement automatic round detection from video
- [ ] Add fighter identification (name/corner assignment)
- [ ] WebSocket real-time updates during processing
- [ ] Docker Compose for one-command deployment
- [ ] PDF/CSV fight report export
- [ ] Multi-camera angle support

<br>

## Tech Stack

- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Next.js 14, React, TypeScript, Tailwind CSS
- **CV Pipeline**: YOLOv8 (Ultralytics), OpenCV, NumPy
- **Tracking**: ByteTrack (built-in YOLOv8), custom identity tracker
- **Scoring**: Custom `FightScorer` with EMA-smoothed metrics

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection
- [FastAPI](https://fastapi.tiangolo.com/) — backend framework
- [Next.js](https://nextjs.org/) — frontend framework
