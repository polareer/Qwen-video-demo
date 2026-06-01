"""Extract first-person assembly video frames for YOLO annotation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract video frames into a YOLO dataset image split.")
    parser.add_argument("--source", required=True, help="Input video path.")
    parser.add_argument("--output-dir", default="datasets/assembly_yolo/images/unlabeled", help="Output image directory.")
    parser.add_argument("--every-sec", type=float, default=0.5, help="Extract one frame every N seconds.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional maximum number of frames. 0 means no limit.")
    parser.add_argument("--prefix", default=None, help="Optional filename prefix. Defaults to the video stem.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video source: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(round(fps * args.every_sec)))
    prefix = args.prefix or source.stem
    frame_index = 0
    saved = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % frame_step == 0:
            timestamp_sec = frame_index / fps
            filename = f"{prefix}_{saved:05d}_{timestamp_sec:07.2f}s.jpg"
            path = output_dir / filename
            cv2.imwrite(str(path), frame)
            saved += 1
            if args.max_frames and saved >= args.max_frames:
                break
        frame_index += 1

    capture.release()
    print(f"saved_frames={saved}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
