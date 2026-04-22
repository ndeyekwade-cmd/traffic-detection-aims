import csv
import os
from ultralytics import YOLO
import cv2

TRAFFIC_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "bus", 5: "truck"}

COLORS_BGR = {
    "car":        (0,   255, 0),
    "truck":      (0,   0,   255),
    "bus":        (255, 165, 0),
    "motorcycle": (255, 0,   255),
    "bicycle":    (255, 255, 0),
    "person":     (0,   165, 255),
}


class YOLODetector:
    def __init__(self, model_path: str = "best.pt"):
        self.model = YOLO(model_path)

    def track_video(
        self,
        video_path: str,
        video_id: str,
        selected_classes: list[int],
        confidence: float,
        output_dir: str,
        progress_callback=None,
    ) -> str:
        os.makedirs(os.path.join(output_dir, video_id), exist_ok=True)
        log_path = os.path.join(output_dir, video_id, "log.csv")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        results_gen = self.model.track(
            source=video_path,
            classes=selected_classes,
            conf=confidence,
            stream=True,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        with open(log_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["video_id", "frame_id", "timestamp", "tracking_id", "class", "confidence", "x1", "y1", "x2", "y2"])

            for frame_id, result in enumerate(results_gen):
                timestamp = round(frame_id / fps, 3)

                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        cls_id   = int(box.cls[0])
                        cls_name = TRAFFIC_CLASSES.get(cls_id, "unknown")
                        conf     = float(box.conf[0])
                        track_id = int(box.id[0]) if box.id is not None else -1
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        writer.writerow([video_id, frame_id, timestamp, track_id, cls_name, round(conf, 3), x1, y1, x2, y2])

                if progress_callback and total_frames > 0:
                    progress_callback(frame_id + 1, total_frames)

        return log_path

    def render_frame(self, video_path: str, frame_id: int, detections: list) -> bytes:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        cap.release()

        if not ok:
            return b""

        if detections:
            for det in detections:
                cls_name = det["class"]
                color    = COLORS_BGR.get(cls_name, (255, 255, 255))
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                track_id = det["tracking_id"]
                conf     = det["confidence"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} #{track_id} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        else:
            cv2.putText(frame, "Aucun objet detecte", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()
