#!/usr/bin/env python3
"""Export YOLOv8-nano to ONNX format.

Usage: gen_yolov8.py <output_path>

Requires: ultralytics
"""

import sys
import os
import shutil


def export_yolov8(out_path: str) -> None:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.export(format="onnx", opset=17, simplify=True, imgsz=640, dynamic=False)
    # ultralytics writes yolov8n.onnx in cwd
    generated = "yolov8n.onnx"
    if os.path.exists(generated):
        shutil.move(generated, out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Saved: {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_path>", file=sys.stderr)
        sys.exit(1)
    export_yolov8(sys.argv[1])
