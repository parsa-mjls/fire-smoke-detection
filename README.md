# fire-smoke-detection
Real-Time Fire &amp; Smoke Detection Pipeline


# 🔥 Fire & Smoke Detection (YOLOv8)

A real-time fire and smoke detector built with **YOLOv8n**, trained on a combination
of public fire/smoke datasets. Supports inference on **images, videos, RTSP streams,
and webcams**, with all thresholds and behavior fully configurable via CLI arguments.

> ⚠️ **Note:** This is a public demo/portfolio version of a fire & smoke detection
> module I built as part of a professional project. Proprietary training data,
> internal tooling, and the full production pipeline are not included here — this
> repo focuses on showcasing the trained model, the inference pipeline, and results.

---

## ✨ Features

- Two-class detection: **fire** and **smoke**
- Single unified script for **image / folder of images / video file / RTSP stream / webcam**
- Fully configurable via CLI: confidence, IoU, image size, device, class filters, etc.
- Optional per-frame/per-image inference timing log
- Optional YOLO-format `.txt` label export
- Live preview window (`--show`) or headless batch processing

---

## 📊 Model & Training

| | |
|---|---|
| **Base model** | YOLOv8n |
| **Classes** | `fire`, `smoke` |
| **Training data** | Combination of fire/smoke detection datasets |
| **Epochs** | ~250 |

### Final metrics (validation)

| Metric | Value (approx.) |
|---|---|
| Precision | ~0.71 |
| Recall | ~0.68 |
| mAP@0.5 | ~0.73 |
| mAP@0.5:0.95 | ~0.41 |

### Training curves

![training metrics](docs/training_metrics.png)

*Box/classification/DFL loss steadily decreasing on both train and validation sets,
with precision, recall, and mAP curves converging after ~150 epochs.*

> Full training logs, dataset composition, and intermediate checkpoints are kept
> private

---

## 📁 Repository Structure

```
fire-smoke-detection/
├── inference.py              # Unified inference script (image/video/RTSP/webcam)
├── requirements.txt
├── weights/
│   └── best.pt                # ⬅ place your trained weights here (not tracked in git)
├── assets/
│   ├── images/                 # Sample input/output demo images
│   └── videos/                 # Sample input/output demo videos (short clips)
├── docs/
│   └── training_metrics.png    # Training curves shown above
├── LICENSE
└── README.md
```

---

## 🚀 Installation

```bash
git clone https://github.com/parsa-mjls/fire-smoke-detection.git
cd fire-smoke-detection

python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Place your trained weights at `weights/best.pt` (or point `--weights` to any path).

---

## 🖥️ Usage

### Single image
```bash
python inference.py --weights weights/best.pt --source assets/images/fire1.jpg
```

### Folder of images
```bash
python inference.py --weights weights/best.pt --source assets/images/
```

### Video file
```bash
python inference.py --weights weights/best.pt --source assets/videos/fire_test.mp4
```

### RTSP stream (live view, no saving)
```bash
python inference.py --weights weights/best.pt \
    --source rtsp://user:pass@192.168.1.10:554/stream1 \
    --show --no-save
```

### Webcam
```bash
python inference.py --weights weights/best.pt --source 0 --show
```

### Custom thresholds / device / output folder
```bash
python inference.py --weights weights/best.pt --source assets/videos/fire_test.mp4 \
    --conf 0.35 --iou 0.5 --device 0 --output runs/predict --log-timing
```

---

## ⚙️ Arguments

| Argument | Default | Description |
|---|---|---|
| `--weights` | *(required)* | Path to trained `.pt` weights |
| `--source` | *(required)* | Image, folder, video path, RTSP URL, or webcam index |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.45` | IoU threshold for NMS |
| `--img-size` | `640` | Inference image size |
| `--max-det` | `300` | Max detections per frame/image |
| `--classes` | `None` | Filter by class id(s), e.g. `--classes 0` |
| `--device` | auto | `cpu`, `0`, `0,1`, etc. |
| `--output` | `runs/predict` | Output directory |
| `--save` / `--no-save` | `--save` | Save annotated output |
| `--save-txt` | off | Save YOLO-format label files |
| `--show` | off | Display live output window |
| `--hide-labels` | off | Hide class labels |
| `--hide-conf` | off | Hide confidence scores |
| `--line-thickness` | `2` | Bounding box line width |
| `--vid-stride` | `1` | Process every Nth frame (video/stream) |
| `--max-frames` | `0` | Cap frames for stream/webcam (0 = unlimited) |
| `--log-timing` | off | Write per-item inference time to `timing_log.txt` |

---

## 🖼️ Sample Results

| Input | Detection |
|---|---|
| ![sample1](assets/Images/input/input2.jpg) | ![sample1 out](assets/Images/output2.jpg) |
| ![sample2](assets/Images/input/input11.jpg) | ![sample2 out](assets/Images/output11.jpg) |
| ![sample3](assets/Images/input/input14.jpg) | ![sample3 out](assets/Images/output14.jpg) |
| ![sample4](assets/Images/input/input15.jpg) | ![sample4 out](assets/Images/output15.jpg) |
| ![sample5](assets/Images/input/input17.jpg) | ![sample5 out](assets/Images/output17.jpg) |
| ![sample6](assets/Images/input/input18.jpg) | ![sample6 out](assets/Images/output18.jpg) |
| ![sample7](assets/Images/input/input20.jpg) | ![sample7 out](assets/Images/output20.jpg) |





A short annotated demo video is available at `assets/videos/demo_output.mp4`.

---

## ⚠️ Limitations

- Trained primarily on outdoor/indoor fire and smoke scenes; performance on
  edge cases (e.g. fog, steam, sunset lighting) may vary.
- Small/distant fire or thin smoke can be missed at low resolution — consider
  increasing `--img-size` for such scenarios.
- Not a certified fire-safety detection system; intended for research/portfolio
  demonstration purposes.

## 🔭 Future Work

- Add temporal smoothing / tracking to reduce flicker on video streams
- Export to ONNX / TensorRT for edge deployment
- Expand dataset diversity (night scenes, industrial environments)

---

## 📄 License

This demo code is released under the [MIT License](LICENSE). The trained
weights are provided as-is for demonstration purposes.

---

## 👤 Author

**Parsa_Majlesi**
[LinkedIn](#) · [GitHub](https://github.com/parsa-mjls) · [Email](#)
