#!/usr/bin/env python3
"""
Fire & Smoke Detection - Unified Inference Script
===================================================

Run a trained YOLOv8 fire/smoke detector on:
  - a single image
  - a folder of images
  - a video file
  - an RTSP stream
  - a webcam

All thresholds and behaviors are controlled via CLI arguments, so no code
edits are required to switch between sources or tune the detector.

Examples
--------
Run on a single image:
    python inference.py --weights weights/best.pt --source assets/images/test1.jpg

Run on a folder of images:
    python inference.py --weights weights/best.pt --source assets/images/

Run on a video file:
    python inference.py --weights weights/best.pt --source assets/videos/fire_test.mp4

Run on an RTSP stream (and display live, without saving):
    python inference.py --weights weights/best.pt --source rtsp://user:pass@192.168.1.10:554/stream1 --show --no-save

Run on the default webcam:
    python inference.py --weights weights/best.pt --source 0 --show

Custom thresholds / device / output folder:
    python inference.py --weights weights/best.pt --source video.mp4 --conf 0.35 --iou 0.5 --device 0 --output runs/predict
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fire & Smoke Detection inference (image / video / RTSP / webcam)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--weights", type=str, required=True,
                         help="Path to the trained YOLOv8 weights (.pt)")
    parser.add_argument("--source", type=str, required=True,
                         help="Image path, folder path, video path, RTSP URL, or webcam index (e.g. 0)")

    # Detection thresholds
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size (pixels)")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per frame/image")
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                         help="Filter by class id(s), e.g. --classes 0  (0=fire, 1=smoke)")

    # Runtime / device
    parser.add_argument("--device", type=str, default="",
                         help="Device to run on, e.g. 'cpu', '0', '0,1'. Empty = auto-select")

    # Output behavior
    parser.add_argument("--output", type=str, default="runs/predict", help="Output directory")
    parser.add_argument("--save", dest="save", action="store_true", default=True,
                         help="Save annotated image(s)/video")
    parser.add_argument("--no-save", dest="save", action="store_false",
                         help="Disable saving output (useful for live RTSP/webcam viewing)")
    parser.add_argument("--save-txt", action="store_true",
                         help="Also save detections in YOLO .txt label format")
    parser.add_argument("--show", action="store_true",
                         help="Display output in a live window")
    parser.add_argument("--hide-labels", action="store_true", help="Hide class labels on output")
    parser.add_argument("--hide-conf", action="store_true", help="Hide confidence scores on output")
    parser.add_argument("--line-thickness", type=int, default=2, help="Bounding box line thickness")

    # Video / stream specific
    parser.add_argument("--vid-stride", type=int, default=1,
                         help="Process every Nth frame for video/stream sources (speed vs. smoothness)")
    parser.add_argument("--max-frames", type=int, default=0,
                         help="Stop after N frames for streams/webcam (0 = unlimited, Ctrl+C to stop)")

    # Logging
    parser.add_argument("--log-timing", action="store_true",
                         help="Write a per-frame/per-image inference-time log (timing_log.txt) to the output dir")

    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def resolve_source_type(source: str) -> str:
    """Classify the input source as 'image', 'image_dir', 'video', or 'stream'."""
    if source.isdigit():
        return "stream"  # webcam index

    if source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return "stream"

    path = Path(source)
    if path.is_dir():
        return "image_dir"

    if path.is_file():
        ext = path.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return "image"
        if ext in VIDEO_EXTENSIONS:
            return "video"

    raise ValueError(f"Could not determine source type for: {source}")


def get_predict_kwargs(args):
    return dict(
        conf=args.conf,
        iou=args.iou,
        imgsz=args.img_size,
        max_det=args.max_det,
        classes=args.classes,
        device=args.device if args.device != "" else None,
        save_txt=args.save_txt,
        line_width=args.line_thickness,
        verbose=False,
    )


def plot_kwargs(args):
    return dict(
        labels=not args.hide_labels,
        conf=not args.hide_conf,
        line_width=args.line_thickness,
    )


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def run_on_single_image(model, image_path, args, out_dir, timing_log):
    print(f"[INFO] Running detection on image: {image_path}")
    start = time.time()
    results = model.predict(source=image_path, **get_predict_kwargs(args))
    elapsed = (time.time() - start) * 1000
    print(f"[INFO] Inference time: {elapsed:.2f} ms")

    if timing_log is not None:
        timing_log.write(f"{os.path.basename(image_path)}\t{elapsed:.4f}\n")

    annotated = results[0].plot(**plot_kwargs(args))

    if args.save:
        save_path = os.path.join(out_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, annotated)
        print(f"[INFO] Saved result to: {save_path}")

    if args.show:
        cv2.imshow("Fire & Smoke Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_on_image_dir(model, dir_path, args, out_dir, timing_log):
    image_files = sorted(
        f for f in os.listdir(dir_path) if Path(f).suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"[WARN] No images found in: {dir_path}")
        return

    print(f"[INFO] Found {len(image_files)} images. Running detection...")
    for img_name in image_files:
        img_path = os.path.join(dir_path, img_name)
        run_on_single_image(model, img_path, args, out_dir, timing_log)


def run_on_video_or_stream(model, source, source_type, args, out_dir, timing_log):
    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save and source_type == "video":
        out_name = Path(source).stem + "_annotated.mp4"
        out_path = os.path.join(out_dir, out_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"[INFO] Saving annotated video to: {out_path}")
    elif args.save and source_type == "stream":
        print("[INFO] --save with a live stream/webcam is not written as a single file "
              "by default (unbounded length). Use --max-frames to cap it, or --no-save "
              "for pure live viewing.")

    frame_idx = 0
    processed = 0
    total_time = 0.0

    print(f"[INFO] Starting inference on {source_type}: {source}  (Ctrl+C to stop)")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % args.vid_stride != 0:
                continue

            start = time.time()
            results = model.predict(source=frame, **get_predict_kwargs(args))
            elapsed = (time.time() - start) * 1000
            total_time += elapsed
            processed += 1

            if timing_log is not None:
                timing_log.write(f"frame_{frame_idx}\t{elapsed:.4f}\n")

            annotated = results[0].plot(**plot_kwargs(args))

            if writer is not None:
                writer.write(annotated)

            if args.show:
                cv2.imshow("Fire & Smoke Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and processed >= args.max_frames:
                print(f"[INFO] Reached --max-frames={args.max_frames}, stopping.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if processed:
        avg_ms = total_time / processed
        print(f"[INFO] Processed {processed} frames | Avg inference time: {avg_ms:.2f} ms "
              f"(~{1000.0 / avg_ms:.1f} FPS)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    if not os.path.isfile(args.weights):
        print(f"[ERROR] Weights file not found: {args.weights}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print(f"[INFO] Loading model: {args.weights}")
    model = YOLO(args.weights)

    timing_log = None
    if args.log_timing:
        timing_path = os.path.join(args.output, "timing_log.txt")
        timing_log = open(timing_path, "w", encoding="utf-8")
        timing_log.write("Item\tInference_Time(ms)\n")

    try:
        source_type = resolve_source_type(args.source)
        print(f"[INFO] Detected source type: {source_type}")

        if source_type == "image":
            run_on_single_image(model, args.source, args, args.output, timing_log)
        elif source_type == "image_dir":
            run_on_image_dir(model, args.source, args, args.output, timing_log)
        else:  # video or stream
            run_on_video_or_stream(model, args.source, source_type, args, args.output, timing_log)

    finally:
        if timing_log is not None:
            timing_log.close()
            print(f"[INFO] Timing log saved to: {os.path.join(args.output, 'timing_log.txt')}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
