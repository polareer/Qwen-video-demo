"""Sweep hand-pose/video offsets and summarize timeline overlap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ergonomics.hand_pose import load_hand_pose_timeline


def video_duration(path: str) -> float:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    capture.release()
    return frame_count / fps if fps else 0.0


def offset_values(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("--step must be greater than 0")
    values: list[float] = []
    current = start
    while current <= end + 1e-9:
        values.append(round(current, 3))
        current += step
    return values


def summarize_offset(video_sec: float, csv_path: str, offset: float) -> dict[str, object]:
    timeline = load_hand_pose_timeline(csv_path, offset_sec=offset)
    sample_video_times = [sample.time - offset for sample in timeline.samples]
    event_video_times = [event.event_time - offset for event in timeline.events if event.event_type == "hole_completed"]
    covered_samples = [time for time in sample_video_times if 0 <= time <= video_sec]
    covered_events = [time for time in event_video_times if 0 <= time <= video_sec]
    sample_coverage = len(covered_samples) / len(sample_video_times) if sample_video_times else 0.0
    event_coverage = len(covered_events) / len(event_video_times) if event_video_times else 0.0
    return {
        "offset_sec": offset,
        "video_duration_sec": round(video_sec, 3),
        "sample_count": len(sample_video_times),
        "sample_covered_count": len(covered_samples),
        "sample_coverage": round(sample_coverage, 3),
        "hole_event_count": len(event_video_times),
        "hole_event_covered_count": len(covered_events),
        "hole_event_coverage": round(event_coverage, 3),
        "covered_event_video_times": [round(time, 3) for time in covered_events],
        "sample_video_time_range": [
            round(min(sample_video_times), 3) if sample_video_times else None,
            round(max(sample_video_times), 3) if sample_video_times else None,
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep hand-pose offset values against a video timeline.")
    parser.add_argument("--source", required=True, help="Video path.")
    parser.add_argument("--hand-pose-csv", required=True, help="Hand trajectory CSV path.")
    parser.add_argument("--start", type=float, default=-10.0, help="Start offset in seconds.")
    parser.add_argument("--end", type=float, default=10.0, help="End offset in seconds.")
    parser.add_argument("--step", type=float, default=1.0, help="Offset step in seconds.")
    parser.add_argument("--top", type=int, default=8, help="Number of best offsets to print.")
    parser.add_argument("--output", default=None, help="Optional JSON output path for all sweep results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    duration_sec = video_duration(args.source)
    results = [
        summarize_offset(duration_sec, args.hand_pose_csv, offset)
        for offset in offset_values(args.start, args.end, args.step)
    ]
    results.sort(
        key=lambda item: (
            float(item["hole_event_coverage"]),
            int(item["hole_event_covered_count"]),
            float(item["sample_coverage"]),
        ),
        reverse=True,
    )
    payload = {
        "source": args.source,
        "hand_pose_csv": args.hand_pose_csv,
        "best_offsets": results[: args.top],
        "all_offsets": results,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_offsets": results[: args.top]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
