"""
Ring Dynamics — Fighter Pose & Knockdown Model Training Pipeline

Fine-tunes YOLOv8-pose for boxing and combat sports, explicitly optimizing
for inverted fighters, knockdowns, and fighters lying on the canvas with
heads positioned at the bottom of the screen and feet at the top.

Key augmentations enabled for knockdown/ground training:
  - flipud = 0.5     (50% vertical flip — teaches network inverted fighter poses)
  - degrees = 25.0   (rotation — handles canvas angles and angled knockdowns)
  - shear = 5.0      (perspective shear on ring canvas)
  - scale = 0.5      (zoom/distance invariance)
"""

import os
import sys
import shutil
import argparse
import random
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO


def extract_augmented_frames(video_path: str, output_dir: str, num_samples: int = 30):
    """
    Extracts frames from a fight video and generates both standard and
    inverted/knockdown augmented frames with corresponding YOLO pose labels.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 100

    step = max(1, total_frames // num_samples)

    images_train = Path(output_dir) / "images" / "train"
    images_val = Path(output_dir) / "images" / "val"
    labels_train = Path(output_dir) / "labels" / "train"
    labels_val = Path(output_dir) / "labels" / "val"

    for p in [images_train, images_val, labels_train, labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    base_model = YOLO("yolov8n-pose.pt")
    saved_count = 0

    frame_idx = 0
    while frame_idx < total_frames and saved_count < num_samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        frame_idx += step
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        res = base_model(frame, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue

        # Extract labels
        lines_orig = []
        lines_inverted = []

        boxes = res.boxes.xywhn.cpu().numpy()
        kpts = res.keypoints.xyn.cpu().numpy() if res.keypoints is not None else None

        for i, box in enumerate(boxes):
            cx, cy, bw, bh = box
            kp_str_orig = []
            kp_str_inv = []

            if kpts is not None and i < len(kpts):
                for kp in kpts[i]:
                    kx, ky = kp[0], kp[1]
                    vis = 2 if (kx > 0 and ky > 0) else 0
                    kp_str_orig.append(f"{kx:.6f} {ky:.6f} {vis}")
                    # Inverted: y -> 1 - y
                    ky_inv = (1.0 - ky) if ky > 0 else 0.0
                    kp_str_inv.append(f"{kx:.6f} {ky_inv:.6f} {vis}")
            else:
                for _ in range(17):
                    kp_str_orig.append("0.0 0.0 0")
                    kp_str_inv.append("0.0 0.0 0")

            lines_orig.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(kp_str_orig))
            cy_inv = 1.0 - cy
            lines_inverted.append(f"0 {cx:.6f} {cy_inv:.6f} {bw:.6f} {bh:.6f} " + " ".join(kp_str_inv))

        if not lines_orig:
            continue

        # Split 80/20 train/val
        is_val = (saved_count % 5 == 0)
        img_dir = images_val if is_val else images_train
        lbl_dir = labels_val if is_val else labels_train

        # Save original frame
        name_orig = f"frame_{saved_count:04d}_orig"
        cv2.imwrite(str(img_dir / f"{name_orig}.jpg"), frame)
        with open(lbl_dir / f"{name_orig}.txt", "w") as f:
            f.write("\n".join(lines_orig) + "\n")

        # Save vertically flipped (knockdown / ground inverted) frame
        frame_inverted = cv2.flip(frame, 0)
        name_inv = f"frame_{saved_count:04d}_inverted"
        cv2.imwrite(str(img_dir / f"{name_inv}.jpg"), frame_inverted)
        with open(lbl_dir / f"{name_inv}.txt", "w") as f:
            f.write("\n".join(lines_inverted) + "\n")

        saved_count += 1

    cap.release()

    # Generate data.yaml
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
"""
    yaml_path = os.path.join(output_dir, "fighter_pose.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Dataset generated at {output_dir}: {saved_count * 2} images (standard + ground-inverted)")
    return yaml_path


def train_model(
    data_yaml: str,
    base_model: str = "yolov8n-pose.pt",
    epochs: int = 5,
    imgsz: int = 640,
    batch: int = 4,
    device: str = "cpu",
    output_weights: str = "backend/models/weights/yolov8n-pose-ground.pt",
):
    """
    Runs YOLOv8-pose training with vertical flip (flipud=0.5) and rotation
    augmentations so the network explicitly learns fighters on the ground / canvas.
    """
    print(f"\n{'='*65}")
    print(f"  Training Combat / Ground Fighter Pose Model")
    print(f"{'='*65}")
    print(f"  Base Model       : {base_model}")
    print(f"  Dataset Config   : {data_yaml}")
    print(f"  Epochs           : {epochs}")
    print(f"  Ground Inversion : flipud=0.5 (Enabled)")
    print(f"  Knockdown Angle  : degrees=25.0 (Enabled)")
    print(f"{'='*65}\n")

    model = YOLO(base_model)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        flipud=0.5,        # 50% probability of vertical flip (inverted fighters)
        fliplr=0.5,        # 50% horizontal flip
        degrees=25.0,      # knockdown rotation
        shear=5.0,         # perspective canvas angles
        scale=0.5,
        save=True,
        verbose=True,
    )

    # Locate best or last weights
    best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = Path(model.trainer.save_dir) / "weights" / "last.pt"

    if best_weights.exists():
        os.makedirs(os.path.dirname(output_weights), exist_ok=True)
        shutil.copy(str(best_weights), output_weights)
        print(f"\n[OK] Model trained successfully! Saved to: {output_weights}")
        # Also copy to backend/yolov8n-pose.pt if desired
        backend_active = "backend/yolov8n-pose.pt"
        shutil.copy(str(best_weights), backend_active)
        print(f"[OK] Updated active backend weights: {backend_active}")
        root_active = "yolov8n-pose.pt"
        shutil.copy(str(best_weights), root_active)
        print(f"[OK] Updated root active weights: {root_active}")
    else:
        print("[!] Training finished, but weights file was not found.")

    return output_weights


def main():
    parser = argparse.ArgumentParser(description="Train / fine-tune YOLOv8-pose for boxing knockdowns & ground fighters")
    parser.add_argument("--video", default="test_clip.mp4", help="Path to fight video for generating dataset")
    parser.add_argument("--dataset-dir", default="backend/storage/fighter_dataset", help="Directory for training dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda:0, mps)")
    parser.add_argument("--base-model", default="yolov8n-pose.pt", help="Base YOLO pose model")
    args = parser.parse_args()

    yaml_path = extract_augmented_frames(args.video, args.dataset_dir, num_samples=20)
    train_model(
        data_yaml=yaml_path,
        base_model=args.base_model,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
