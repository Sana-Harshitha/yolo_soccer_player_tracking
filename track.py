import cv2
import os
from ultralytics import YOLO
from pathlib import Path
import numpy as np
from boxmot.trackers.strongsort.strongsort import StrongSort

def run_tracker(video_path, yolo_weights, reid_weights):
    # Load YOLOv5/YOLOv8 model
    model = YOLO(yolo_weights)
    print("YOLO model loaded.")

    # Initialize StrongSORT tracker
    tracker = StrongSort(
    reid_weights=Path(reid_weights),
    device='cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu',
    half=False,
    max_age=60,         #  Players disappear/reappear across frames
    n_init=2,           #  Confirm tracks faster (helps with quick movement)
    max_iou_dist=0.9,   # More forgiving on bounding box overlaps
    max_cos_dist=0.15,  #  Stronger appearance matching
    nn_budget=150,      # Allow more embeddings to build stable identity
    mc_lambda=0.95,     #  Slightly reduce motion consistency weight
    ema_alpha=0.95      # Smooth out embedding changes (slow jitter)
    )
    print(" StrongSORT initialized.")

    # Open video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs('outputs', exist_ok=True)
    out = cv2.VideoWriter("outputs/strongsort_tracking.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = model(frame, verbose=False)[0]
        detections = []

        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            if int(cls) == 2:  # Only class 2: soccer players
                x1, y1, x2, y2 = map(int, box.tolist())
                detections.append([x1, y1, x2, y2, float(conf), int(cls)])

        # Convert to numpy array with shape [N, 6]
        if len(detections) > 0:
            dets_np = cv2.UMat(np.array(detections, dtype=np.float32)).get()
        else:
            dets_np = np.empty((0, 6), dtype=np.float32)

        # Update tracker
        tracks = tracker.update(dets_np, frame)

        # Draw track IDs
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls, _ = map(int, track)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Write & show
        out.write(frame)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(" Tracking complete. Saved to outputs/strongsort_tracking.mp4")

if __name__ == "__main__":
    run_tracker(
        video_path="videos/15sec_input_720p.mp4",
        yolo_weights="weights/soccer_yolov5_custom.pt",
        reid_weights="weights/osnet_x0_25_msmt17.pt"
    )
