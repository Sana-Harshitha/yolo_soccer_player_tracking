# detect.py

import cv2
from ultralytics import YOLO
import os

def load_model(model_path):
    print(f" Loading model from {model_path}")
    return YOLO(model_path)

def process_video(model, video_path, output_path, conf_thresh=0.3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" Could not open video: {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf_thresh, verbose=False)
        annotated = results[0].plot()

        out.write(annotated)
        cv2.imshow("Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if frame_id % 50 == 0:
            print(f"Processed {frame_id} frames...")

        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Detection complete. Saved to: {output_path}")

def main():
    video_path = "videos/tacticam.mp4"
    model_path = "model/soccer_yolov5_custom.pt"
    output_path = "outputs/tacticam_annotated.mp4"

    model = load_model(model_path)
    process_video(model, video_path, output_path)

if __name__ == "__main__":
    main()
