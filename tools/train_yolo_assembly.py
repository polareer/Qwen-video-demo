"""Train a YOLO model for first-person manual assembly detection."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO for first-person assembly targets.")
    parser.add_argument("--data", default="configs/assembly_yolo.yaml", help="YOLO data yaml.")
    parser.add_argument("--model", default="outputs/weights/yolo11n.pt", help="Base YOLO weight or model yaml.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="0, cpu, auto, etc.")
    parser.add_argument("--project", default="outputs/yolo_train")
    parser.add_argument("--name", default="assembly_yolo11n")
    parser.add_argument("--dry-run", action="store_true", help="Only validate dataset and environment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(tempfile.gettempdir(), "qwen_video_demo_ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", os.environ["YOLO_CONFIG_DIR"])
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    dataset_info = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = (data_yaml.parent / dataset_info["path"]).resolve()
    train_images = dataset_root / dataset_info["train"]
    val_images = dataset_root / dataset_info["val"]
    train_labels = dataset_root / "labels" / "train"
    val_labels = dataset_root / "labels" / "val"

    train_label_count = _count_files(train_labels, "*.txt")
    val_label_count = _count_files(val_labels, "*.txt")
    train_image_count = _count_images(train_images)
    val_image_count = _count_images(val_images)

    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
    print(f"dataset_root={dataset_root}")
    print(f"train_images={train_image_count}, train_labels={train_label_count}")
    print(f"val_images={val_image_count}, val_labels={val_label_count}")
    print(f"classes={dataset_info.get('names')}")

    if train_image_count == 0 or val_image_count == 0:
        raise RuntimeError("Dataset images are missing. Put labeled images into images/train and images/val.")
    if train_label_count == 0 or val_label_count == 0:
        raise RuntimeError("YOLO label txt files are missing. Annotate images before training.")

    if args.dry_run:
        print("dry_run_ok=true")
        return

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=0,
        pretrained=True,
    )


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp") for _ in path.glob(suffix))


def _count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


if __name__ == "__main__":
    main()
