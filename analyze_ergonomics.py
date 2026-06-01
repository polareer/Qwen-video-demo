"""CLI for first-person ergonomic risk analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def build_analyzer(args: argparse.Namespace) -> ErgonomicAnalyzer:
    try:
        from ergonomics import ErgonomicAnalyzer, OllamaVisionClient
        from ergonomics.analyzer import AnalyzerConfig
        from ergonomics.hand_pose import load_hand_pose_timeline
        from ergonomics.tasks import TaskRecognizer
        from ergonomics.vision import FrameAnalyzer
    except ModuleNotFoundError as exc:
        if exc.name == "cv2":
            raise RuntimeError("Missing dependency: opencv-python. Install with: pip install -r requirements.txt") from exc
        raise

    config_data = load_config(args.config)
    ergonomics = config_data.get("ergonomics", {})
    detector = config_data.get("detector", {})
    ollama = config_data.get("ollama", {})
    analyzer_config = AnalyzerConfig(
        analysis_fps=float(args.analysis_fps or ergonomics.get("analysis_fps", 10.0)),
        window_seconds=float(ergonomics.get("window_seconds", 1.0)),
        keyframes_per_event=int(ergonomics.get("keyframes_per_event", 3)),
        min_event_gap_seconds=float(ergonomics.get("min_event_gap_seconds", 3.0)),
        visibility_threshold=float(ergonomics.get("visibility_threshold", 0.55)),
        reachability_threshold=float(ergonomics.get("reachability_threshold", 0.45)),
        low_light_threshold=float(ergonomics.get("low_light_threshold", 0.28)),
        blur_threshold=float(ergonomics.get("blur_threshold", 0.25)),
        center_offset_threshold=float(ergonomics.get("center_offset_threshold", 0.72)),
        occlusion_threshold=float(ergonomics.get("occlusion_threshold", 0.35)),
        use_ollama=not args.no_ollama and bool(ollama.get("enabled", True)),
    )
    frame_analyzer = FrameAnalyzer(
        yolo_model_path=args.yolo_model or detector.get("yolo_model_path"),
        target_label=args.target_label or detector.get("target_label"),
        use_mediapipe_hands=bool(detector.get("use_mediapipe_hands", True)),
        device=args.device or detector.get("device", "auto"),
        detector_confidence=float(args.detector_conf or detector.get("confidence", 0.25)),
        detector_imgsz=int(args.detector_imgsz or detector.get("imgsz", 640)),
    )
    task_recognizer = TaskRecognizer.from_yaml(args.task_template)
    hand_pose_timeline = load_hand_pose_timeline(args.hand_pose_csv, offset_sec=float(args.hand_pose_offset_sec))
    ollama_client = None
    if analyzer_config.use_ollama:
        ollama_client = OllamaVisionClient(
            base_url=args.ollama_url or ollama.get("base_url", "http://127.0.0.1:11434"),
            model=args.ollama_model or ollama.get("model", "qwen2.5vl:7b"),
            timeout_seconds=int(ollama.get("timeout_seconds", 60)),
        )
    return ErgonomicAnalyzer(
        output_dir=args.output_dir or ergonomics.get("output_dir", "./outputs/ergonomics"),
        config=analyzer_config,
        frame_analyzer=frame_analyzer,
        task_recognizer=task_recognizer,
        hand_pose_timeline=hand_pose_timeline,
        ollama_client=ollama_client,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze first-person video for ergonomic visibility/reachability risks.")
    parser.add_argument("--source", required=True, help="Video file path, camera index, or RTSP URL.")
    parser.add_argument("--config", default="configs/ergonomics.yaml", help="YAML config path.")
    parser.add_argument("--output-dir", default=None, help="Directory for reports and keyframes.")
    parser.add_argument("--analysis-fps", type=float, default=None, help="Frame analysis rate.")
    parser.add_argument("--target-label", default=None, help="Preferred target label from detector output.")
    parser.add_argument("--task-template", default="configs/task_templates.yaml", help="YAML SOP task template path.")
    parser.add_argument("--hand-pose-csv", default=None, help="Optional hand trajectory CSV exported from the AR capture tool.")
    parser.add_argument("--hand-pose-offset-sec", type=float, default=0.0, help="Manual offset: hand pose time = video time + offset.")
    parser.add_argument("--yolo-model", default=None, help="Optional Ultralytics YOLO model path.")
    parser.add_argument("--device", default=None, help="Detector device: auto, cpu, 0, cuda:0, etc.")
    parser.add_argument("--detector-conf", type=float, default=None, help="YOLO detector confidence threshold.")
    parser.add_argument("--detector-imgsz", type=int, default=None, help="YOLO detector image size.")
    parser.add_argument("--ollama-url", default=None, help="Ollama base URL.")
    parser.add_argument("--ollama-model", default=None, help="Ollama vision model name.")
    parser.add_argument("--no-ollama", action="store_true", help="Disable Qwen2.5-VL explanation and use local rule explanation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source: str | int = int(args.source) if args.source.isdigit() else args.source
    analyzer = build_analyzer(args)
    result = analyzer.analyze_video(source)
    print(json.dumps({"report_path": result["report_path"], "event_count": result["summary"]["event_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
