---
title: Traffic Detection AIMS
emoji: 🚗
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Road Traffic Object Detection & Tracking

**AIMS Sénégal — Computer Vision Project 2 — April 2026**  
**Group:** Ndeye Khady Wade · Adama Telly Ba · Ismaila Tikome Nana
---

## Overview

A real-time web application for detecting, tracking, and counting road-traffic objects across multiple video scenes. Built with **YOLOv8** (fine-tuned) + **ByteTrack**, served through a **FastAPI** backend and a dark-themed HTML/JS frontend.

The system processes traffic videos frame by frame, assigns unique persistent IDs to each detected object, counts unique vehicles/pedestrians across the full scene, and generates detailed logs for analysis.

---

## Features

- **Detection & Tracking** — YOLOv8 fine-tuned model + ByteTrack algorithm
- **6 traffic classes** — car, truck, bus, motorcycle, bicycle, person
- **Unique object counting** — counts distinct tracks, not raw per-frame detections
- **Direction detection** — dominant movement direction per class (↑ ↓ ← →)
- **Temporal distribution** — traffic intensity chart over time (objects per 5-second bucket)
- **Multi-scene dashboard** — compare stats across all analyzed videos
- **Detailed CSV logs** — timestamp, class, bounding box, tracking ID per detection
- **Web interface** — drag & drop upload, class selection, confidence slider, live frame viewer

---

## Project Structure

```
cv_project_2/
├── main.py          # FastAPI backend (7 endpoints)
├── detector.py      # YOLODetector class (YOLOv8 + ByteTrack)
├── index.html       # Frontend (HTML/CSS/JS vanilla)
├── requirements.txt # Python dependencies
├── LICENSE
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.10+
- `best.pt` — fine-tuned YOLOv8 model (place in project root)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/traffic-detection-aims.git
cd traffic-detection-aims

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your model weights
# Place best.pt in the project root

# 4. Start the server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 5. Open in browser
# http://localhost:8000
```

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET`  | `/` | Web interface |
| `POST` | `/upload` | Upload a video file |
| `POST` | `/analyser` | Run detection + tracking |
| `GET`  | `/progress/{video_id}` | Analysis progress (%) |
| `GET`  | `/frame/{video_id}/{frame_id}` | Annotated frame (JPEG) |
| `GET`  | `/frame_count/{video_id}` | Total number of frames |
| `GET`  | `/stats/{video_id}` | Unique objects, avg duration, direction |
| `GET`  | `/temporal/{video_id}` | Traffic intensity over time |
| `GET`  | `/dashboard` | All videos stats (multi-scene) |
| `GET`  | `/logs/{video_id}` | Full CSV logs as JSON |

---

## Log Schema (CSV)

All detections are saved to `data/output/{video_id}/log.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | string | Unique video identifier |
| `frame_id` | int | Frame number |
| `timestamp` | float | Time in seconds |
| `tracking_id` | int | Unique persistent object ID |
| `class` | string | Detected class name |
| `confidence` | float | Detection confidence score |
| `x1, y1, x2, y2` | int | Bounding box coordinates |

---

## Detected Classes

| Class ID | Name | Color |
|----------|------|-------|
| 0 | person | orange |
| 1 | bicycle | cyan |
| 2 | car | green |
| 3 | motorcycle | purple |
| 4 | bus | orange |
| 5 | truck | red |

---

## Tech Stack

- **Backend:** FastAPI, Uvicorn, OpenCV, Ultralytics YOLOv8
- **Tracking:** ByteTrack (via Ultralytics)
- **Frontend:** HTML5, CSS3, JavaScript (vanilla), Chart.js
- **Data:** CSV logs, in-memory frame cache

---

## On Kaggle (GPU T4)

```python
import subprocess, threading

def run():
    subprocess.run(["python", "-m", "uvicorn", "main:app",
                    "--host", "0.0.0.0", "--port", "8000"])

threading.Thread(target=run, daemon=True).start()
# Then expose port 8000 with ngrok
```

---

## License

MIT License — see [LICENSE](LICENSE)
