import csv
import os
import shutil
import uuid
from collections import defaultdict
from pathlib import Path

import cv2
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

from detector import TRAFFIC_CLASSES, YOLODetector

app = FastAPI(title="Traffic Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = "data/uploads"
OUTPUT_DIR = "data/output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

detector = YOLODetector("best.pt")
progress_store: dict[str, dict] = {}
frame_cache: dict[str, dict[int, list]] = {}


def load_log_cache(video_id: str):
    if video_id in frame_cache:
        return
    log_path = Path(OUTPUT_DIR) / video_id / "log.csv"
    if not log_path.exists():
        return
    cache: dict[int, list] = defaultdict(list)
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            fid = int(row["frame_id"])
            cache[fid].append({
                "class":       row["class"],
                "tracking_id": int(row["tracking_id"]),
                "confidence":  float(row["confidence"]),
                "x1": int(row["x1"]), "y1": int(row["y1"]),
                "x2": int(row["x2"]), "y2": int(row["y2"]),
            })
    frame_cache[video_id] = dict(cache)


def compute_direction(dx: int, dy: int) -> str:
    if abs(dx) < 15 and abs(dy) < 15:
        return "stationnaire"
    if abs(dx) > abs(dy):
        return "→" if dx > 0 else "←"
    return "↓" if dy > 0 else "↑"


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path("index.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".mp4"
    save_path = os.path.join(UPLOAD_DIR, f"{video_id}{ext}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"video_id": video_id, "filename": file.filename, "path": save_path}


@app.post("/analyser")
async def analyser(
    background_tasks: BackgroundTasks,
    video_id: str = Form(...),
    classes: str = Form("2,3,5,7"),
    confidence: float = Form(0.3),
):
    video_files = list(Path(UPLOAD_DIR).glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="Vidéo introuvable")

    selected_classes = [int(c.strip()) for c in classes.split(",") if c.strip().isdigit()]
    selected_classes = [c for c in selected_classes if c in TRAFFIC_CLASSES]
    if not selected_classes:
        raise HTTPException(status_code=400, detail="Aucune classe valide sélectionnée")

    progress_store[video_id] = {"done": 0, "total": 0, "status": "running"}
    frame_cache.pop(video_id, None)

    def run_detection():
        try:
            detector.track_video(
                video_path=str(video_files[0]),
                video_id=video_id,
                selected_classes=selected_classes,
                confidence=confidence,
                output_dir=OUTPUT_DIR,
                progress_callback=lambda done, total: progress_store[video_id].update(
                    {"done": done, "total": total}
                ),
            )
            progress_store[video_id]["status"] = "done"
        except Exception as e:
            progress_store[video_id]["status"] = f"error: {e}"

    background_tasks.add_task(run_detection)
    return {"video_id": video_id, "message": "Analyse démarrée"}


@app.get("/progress/{video_id}")
async def get_progress(video_id: str):
    info = progress_store.get(video_id, {"done": 0, "total": 0, "status": "unknown"})
    pct = round(info["done"] / info["total"] * 100, 1) if info["total"] > 0 else 0
    return {**info, "percent": pct}


@app.get("/frame/{video_id}/{frame_id}")
async def get_frame(video_id: str, frame_id: int):
    video_files = list(Path(UPLOAD_DIR).glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="Vidéo introuvable")
    load_log_cache(video_id)
    detections = frame_cache.get(video_id, {}).get(frame_id, [])
    jpeg = detector.render_frame(str(video_files[0]), frame_id, detections)
    if not jpeg:
        raise HTTPException(status_code=404, detail="Frame introuvable")
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/frame_count/{video_id}")
async def frame_count(video_id: str):
    video_files = list(Path(UPLOAD_DIR).glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="Vidéo introuvable")
    cap = cv2.VideoCapture(str(video_files[0]))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"video_id": video_id, "frame_count": count}


@app.get("/stats/{video_id}")
async def get_stats(video_id: str):
    log_path = Path(OUTPUT_DIR) / video_id / "log.csv"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Logs introuvables")

    track_times:     dict[tuple, list] = defaultdict(list)
    track_positions: dict[tuple, list] = defaultdict(list)
    class_tracks:    dict[str, set]    = defaultdict(set)

    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["class"], row["tracking_id"])
            track_times[key].append(float(row["timestamp"]))
            class_tracks[row["class"]].add(row["tracking_id"])
            cx = (int(row["x1"]) + int(row["x2"])) // 2
            cy = (int(row["y1"]) + int(row["y2"])) // 2
            track_positions[key].append((int(row["frame_id"]), cx, cy))

    stats = {}
    for cls_name, track_ids in class_tracks.items():
        durations  = []
        directions = []
        for tid in track_ids:
            key   = (cls_name, tid)
            times = track_times[key]
            if len(times) > 1:
                durations.append(max(times) - min(times))
            positions = sorted(track_positions[key])
            if len(positions) >= 2:
                fx, fy = positions[0][1], positions[0][2]
                lx, ly = positions[-1][1], positions[-1][2]
                directions.append(compute_direction(lx - fx, ly - fy))

        dominant = max(set(directions), key=directions.count) if directions else "—"
        stats[cls_name] = {
            "unique_objects":     len(track_ids),
            "avg_duration_sec":   round(sum(durations) / len(durations), 2) if durations else 0.0,
            "dominant_direction": dominant,
        }

    return {"video_id": video_id, "stats": stats}


@app.get("/temporal/{video_id}")
async def get_temporal(video_id: str, interval: int = 5):
    log_path = Path(OUTPUT_DIR) / video_id / "log.csv"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Logs introuvables")

    buckets: dict[int, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            bucket = int(float(row["timestamp"]) // interval) * interval
            buckets[bucket][row["class"]].add(row["tracking_id"])

    data = {
        f"{b}s": {cls: len(ids) for cls, ids in classes.items()}
        for b, classes in sorted(buckets.items())
    }
    return {"video_id": video_id, "interval": interval, "data": data}


@app.get("/dashboard")
async def get_dashboard():
    videos = []
    for video_dir in sorted(Path(OUTPUT_DIR).iterdir()):
        if not video_dir.is_dir():
            continue
        log_path = video_dir / "log.csv"
        if not log_path.exists():
            continue

        video_id     = video_dir.name
        class_tracks: dict[str, set] = defaultdict(set)
        max_frame    = 0

        with open(log_path, newline="") as f:
            for row in csv.DictReader(f):
                class_tracks[row["class"]].add(row["tracking_id"])
                max_frame = max(max_frame, int(row["frame_id"]))

        video_files = list(Path(UPLOAD_DIR).glob(f"{video_id}.*"))
        filename    = video_files[0].name if video_files else video_id

        videos.append({
            "video_id":     video_id,
            "filename":     filename,
            "total_frames": max_frame + 1,
            "stats":        {cls: len(ids) for cls, ids in class_tracks.items()},
        })

    return {"videos": videos}


@app.get("/logs/{video_id}")
async def get_logs(video_id: str):
    log_path = Path(OUTPUT_DIR) / video_id / "log.csv"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Logs introuvables")
    with open(log_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {"video_id": video_id, "count": len(rows), "logs": rows}
