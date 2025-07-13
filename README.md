# Soccer Player Re-Identification using YOLOv5 + StrongSORT

This project performs real-time soccer player detection and tracking using a custom-trained YOLOv5 model integrated with StrongSORT for robust player re-identification (ReID) in match footage.

---

##  Objective

To accurately detect and consistently track soccer players in video footage, preserving unique IDs even under occlusions, re-entries, and motion blur.  
This project was developed as part of the **Soccer Player Re-Identification Assignment**.

---

##  Project Structure

```
Yolov5_StrongSORT_OSNet/
├── track.py                        # Main script to run tracking
├── model/
│   └── soccer_yolov5_custom.pt     # Custom YOLOv5 model trained on player dataset
├── weights/
│   └── osnet_x0_25_market1501.pt   # StrongSORT ReID model
├── videos/
│   └── 15sec_input_720p.mp4        # Input test video
├── outputs/
│   └── strongsort_tracking.mp4     # Output video with tracking
├── boxmot/
│   └── trackers/
│       └── strongsort/             # StrongSORT tracking code
│       └── bytetrack/              # (Optional) ByteTrack code
│   └── appearance/reid/            # ReID loader
│       └── auto_backend.py
├── README.md                       # This documentation
```

---

##  Setup Instructions

### 1. Clone and Install

```bash
git clone https://github.com/<your_repo>/Yolov5_StrongSORT_OSNet.git
cd Yolov5_StrongSORT_OSNet
pip install -r requirements.txt
```

### 2. Install Dependencies

```bash
# Install in editable mode with extras
pip install -e .[yolo,test,dev]
```

- Uses `pyproject.toml`
- No need for requirements.txt
- `-e .` = editable mode

### 3. Place Required Files

- `model/soccer_yolov5_custom.pt` → your trained YOLOv5 player detector
- `weights/osnet_x0_25_market1501.pt` → ReID model
- `videos/15sec_input_720p.mp4` → input test video

---

### 4. Run Tracker

```bash
python track.py
```

The output video with tracking overlays will be saved in:

```
outputs/strongsort_tracking.mp4
```

---

##  Tracking Parameters (Tuned for Soccer Game Footage)

```python
tracker = StrongSort(
    reid_weights=Path(reid_weights),
    device='cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu',
    half=False,
    max_age=60,              # Allow longer time for occlusion
    n_init=2,                # Faster ID confirmation
    max_iou_dist=0.85,       # More forgiving for fast motion
    max_cos_dist=0.2,        # Appearance embedding threshold
    nn_budget=150,           # ReID memory
    mc_lambda=0.995,
    ema_alpha=0.95
)
```

---

##  Approach & Methodology

- **YOLOv5**: Custom-trained to detect only soccer players (class ID = 2).
- **StrongSORT**: Used for appearance-based ReID tracking.
- **Filtered Detections**: Only players passed to tracker.
- **Clean ID Overlay**: Green bounding boxes + ID labels.
- **Saved Output**: Tracked video written frame-by-frame.

---


### Key Results

- ReID model struggled under player re-entry and visual ambiguity.
- Bounding box jitter and motion blur caused partial ID switches.
- Appearance-based ReID worked reasonably well for short segments.

---

##  Challenges

- Inconsistent IDs on fast re-entry of players.
- Minor jersey folds caused different embeddings.
- Class-based filtering was essential for accurate player-only tracking.

---

##  Future Work

If time/resources permitted:

- Fine-tune OSNet on soccer dataset (or use a stronger ReID model).
- OCR jersey numbers and combine with ReID for stronger identity.
- Apply temporal smoothing or Kalman filtering on ID confidence.
- Consider merging ByteTrack + StrongSORT for optimal results.

---

