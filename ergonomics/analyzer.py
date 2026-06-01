"""Video-level ergonomic analysis pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import cv2

from .hand_pose import HandPoseTimeline
from .ollama_client import OllamaVisionClient
from .schemas import ErgonomicMetrics, ErgonomicRiskEvent, Evidence, HandPoseMetrics, QwenExplanation, TargetState, TaskState
from .tasks import TaskRecognizer, UNKNOWN_TASK
from .vision import FrameAnalysis, FrameAnalyzer


@dataclass
class AnalyzerConfig:
    analysis_fps: float = 10.0
    window_seconds: float = 1.0
    keyframes_per_event: int = 3
    min_event_gap_seconds: float = 3.0
    visibility_threshold: float = 0.55
    reachability_threshold: float = 0.45
    low_light_threshold: float = 0.28
    blur_threshold: float = 0.25
    center_offset_threshold: float = 0.72
    occlusion_threshold: float = 0.35
    use_ollama: bool = True


class ErgonomicAnalyzer:
    """Runs first-person task recognition and ergonomic risk analysis over a video."""

    def __init__(
        self,
        output_dir: str,
        config: AnalyzerConfig | None = None,
        frame_analyzer: FrameAnalyzer | None = None,
        task_recognizer: TaskRecognizer | None = None,
        hand_pose_timeline: HandPoseTimeline | None = None,
        ollama_client: OllamaVisionClient | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.keyframe_dir = self.output_dir / "keyframes"
        self.config = config or AnalyzerConfig()
        self.frame_analyzer = frame_analyzer or FrameAnalyzer()
        self.task_recognizer = task_recognizer or TaskRecognizer([])
        self.hand_pose_timeline = hand_pose_timeline
        self.ollama_client = ollama_client

    def analyze_video(self, source: str | int) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keyframe_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video source: {source}")

        native_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_step = max(1, int(round(native_fps / self.config.analysis_fps)))
        video_id = f"camera_{source}" if isinstance(source, int) else Path(source).stem
        frame_index = 0
        window: list[tuple[FrameAnalysis, Any]] = []
        events: list[ErgonomicRiskEvent] = []
        task_windows: list[dict[str, Any]] = []
        last_event_time: dict[str, float] = {}

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_step != 0:
                frame_index += 1
                continue
            timestamp_sec = frame_index / native_fps
            analysis = self.frame_analyzer.analyze(frame, frame_index, timestamp_sec)
            window.append((analysis, frame.copy()))
            if self._window_ready(window):
                self._process_window(video_id, window, events, task_windows, last_event_time)
                window = []
            frame_index += 1

        if window:
            self._process_window(video_id, window, events, task_windows, last_event_time)

        capture.release()
        task_timeline = self._merge_task_timeline(task_windows)
        report = {
            "video_id": video_id,
            "source": source,
            "task_timeline": task_timeline,
            "events": [event.to_dict() for event in events],
            "summary": {
                "event_count": len(events),
                "risk_types": sorted({event.risk_type for event in events}),
                "task_coverage": self._task_coverage(task_windows),
                "hand_pose_available": self.hand_pose_timeline is not None,
                "hand_pose_coverage": self._hand_pose_coverage(task_windows),
                "hole_event_count": self.hand_pose_timeline.event_count if self.hand_pose_timeline is not None else 0,
                "high_risk_event_count": sum(1 for event in events if event.risk_level == "high"),
                "needs_human_review": any(event.qwen_explanation.needs_human_review for event in events),
            },
        }
        report_path = self.output_dir / f"{video_id}_ergonomic_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"report_path": str(report_path), **report}

    def _process_window(
        self,
        video_id: str,
        window: list[tuple[FrameAnalysis, Any]],
        events: list[ErgonomicRiskEvent],
        task_windows: list[dict[str, Any]],
        last_event_time: dict[str, float],
    ) -> None:
        task_state = self.task_recognizer.recognize([item[0] for item in window])
        hand_pose_metrics = self._hand_pose_window_metrics(window)
        task_windows.append(self._task_window(task_state, window, hand_pose_metrics))
        event = self._event_from_window(video_id, window, len(events) + 1, last_event_time, task_state, hand_pose_metrics)
        if event is None:
            return
        if self.config.use_ollama and self.ollama_client is not None:
            event.qwen_explanation = self.ollama_client.explain_event(event)
        else:
            event.qwen_explanation = self._local_explanation(event)
        events.append(event)
        last_event_time[event.risk_type] = window[-1][0].timestamp_sec

    def _window_ready(self, window: list[tuple[FrameAnalysis, Any]]) -> bool:
        return bool(window) and (window[-1][0].timestamp_sec - window[0][0].timestamp_sec) >= self.config.window_seconds

    def _event_from_window(
        self,
        video_id: str,
        window: list[tuple[FrameAnalysis, Any]],
        next_index: int,
        last_event_time: dict[str, float],
        task_state: TaskState,
        hand_pose_metrics: HandPoseMetrics,
    ) -> ErgonomicRiskEvent | None:
        analyses = [item[0] for item in window]
        risk_type = self._risk_type(analyses, hand_pose_metrics)
        if risk_type is None:
            return None
        start = analyses[0].timestamp_sec
        end = analyses[-1].timestamp_sec
        if start - last_event_time.get(risk_type, -999.0) < self.config.min_event_gap_seconds:
            return None
        keyframes = self._save_keyframes(video_id, next_index, window)
        target = analyses[-1].target
        metrics = ErgonomicMetrics(
            visibility_score=round(mean(item.visibility_score for item in analyses), 3),
            reachability_score=round(mean(item.reachability_score for item in analyses), 3),
            occlusion_duration_sec=round(
                sum(1 for item in analyses if item.occlusion_ratio >= self.config.occlusion_threshold)
                / self.config.analysis_fps,
                2,
            ),
            center_offset=round(mean(item.center_offset for item in analyses), 3),
            brightness=round(mean(item.brightness for item in analyses), 3),
            sharpness=round(mean(item.sharpness for item in analyses), 3),
        )
        return ErgonomicRiskEvent(
            event_id=f"erg_{next_index:05d}",
            video_id=video_id,
            time_range=[self._format_time(start), self._format_time(end)],
            risk_type=risk_type,
            risk_level=self._risk_level(metrics, hand_pose_metrics),
            task_id=task_state.task_id,
            task_name=task_state.task_name,
            task_confidence=task_state.confidence,
            task_phase=task_state.task_phase,
            target=TargetState(name=target.label, visibility=self._visibility_state(metrics), bbox=target.bbox),
            metrics=metrics,
            evidence=Evidence(keyframes=keyframes),
            hand_pose_metrics=hand_pose_metrics,
        )

    def _risk_type(self, analyses: list[FrameAnalysis], hand_pose_metrics: HandPoseMetrics) -> str | None:
        visibility = mean(item.visibility_score for item in analyses)
        reachability = mean(item.reachability_score for item in analyses)
        brightness = mean(item.brightness for item in analyses)
        sharpness = mean(item.sharpness for item in analyses)
        center_offset = mean(item.center_offset for item in analyses)
        occlusion = mean(item.occlusion_ratio for item in analyses)
        if brightness < self.config.low_light_threshold:
            return "visibility_low_light"
        if sharpness < self.config.blur_threshold:
            return "visibility_blur"
        if occlusion >= self.config.occlusion_threshold:
            return "visibility_occlusion"
        if center_offset >= self.config.center_offset_threshold:
            return "visibility_edge"
        if hand_pose_metrics.hand_pose_matched:
            if visibility < self.config.visibility_threshold and (hand_pose_metrics.hand_speed_mm_s or 0.0) >= 40.0:
                return "operation_without_visibility"
            if (hand_pose_metrics.reach_distance_mm or 0.0) >= 650.0:
                return "reachability_overextended"
            if hand_pose_metrics.stability_score is not None and hand_pose_metrics.stability_score < 0.35:
                return "operation_unstable"
        elif self.hand_pose_timeline is not None:
            return "pose_missing"
        if visibility < self.config.visibility_threshold:
            return "visibility_limited"
        if reachability < self.config.reachability_threshold:
            return "reachability_limited"
        return None

    def _save_keyframes(self, video_id: str, event_index: int, window: list[tuple[FrameAnalysis, Any]]) -> list[str]:
        step = max(1, len(window) // self.config.keyframes_per_event)
        selected = window[::step][: self.config.keyframes_per_event]
        paths: list[str] = []
        for analysis, frame in selected:
            annotated = self._draw_annotation(frame, analysis)
            filename = f"{video_id}_erg_{event_index:05d}_frame_{analysis.frame_index}.jpg"
            path = self.keyframe_dir / filename
            cv2.imwrite(str(path), annotated)
            paths.append(str(path))
        return paths

    def _draw_annotation(self, frame: Any, analysis: FrameAnalysis) -> Any:
        target_color = (255, 82, 0)
        hand_color = (0, 180, 255)
        x1, y1, x2, y2 = analysis.target.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), target_color, 2)
        cv2.putText(frame, analysis.target.label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, target_color, 2)
        for hand in analysis.hands:
            hx1, hy1, hx2, hy2 = hand.bbox
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), hand_color, 2)
            cv2.putText(frame, "hand", (hx1, max(20, hy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
        label = f"vis={analysis.visibility_score:.2f} reach={analysis.reachability_score:.2f}"
        cv2.putText(frame, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 82, 255), 2)
        return frame

    def _local_explanation(self, event: ErgonomicRiskEvent) -> QwenExplanation:
        explanations = {
            "visibility_low_light": "画面亮度偏低，关键目标状态难以稳定确认。",
            "visibility_blur": "画面清晰度偏低，可能由快速运动、相机晃动或对焦不稳定导致。",
            "visibility_occlusion": "操作目标与手部或工具重叠较高，关键接触点存在被遮挡风险。",
            "visibility_edge": "关键目标长期位于视野边缘，操作者可能无法稳定观察。",
            "visibility_limited": "目标可视性综合评分偏低，建议调整视角或补光。",
            "reachability_limited": "手部与目标空间关系不稳定，疑似存在可达性或操作姿态受限。",
            "reachability_overextended": "手部相对头显距离偏大，疑似存在过伸或远距离操作。",
            "operation_unstable": "手部轨迹稳定性偏低，操作点附近可能存在反复试探或抖动。",
            "pose_missing": "该窗口未匹配到手部轨迹数据，无法用空间位姿证据复核操作。",
            "operation_without_visibility": "视频可视性偏低，但手部轨迹显示仍在持续操作，存在证据不足的操作风险。",
        }
        recommendations = {
            "visibility_low_light": "增加局部补光或调整头戴相机角度。",
            "visibility_blur": "降低头部晃动，检查相机固定和对焦状态。",
            "visibility_occlusion": "调整手部或工具角度，确保接触点可见后继续操作。",
            "visibility_edge": "调整站位或头部视角，让目标进入画面中心区域。",
            "visibility_limited": "复核目标可见性，必要时切换补盲视角。",
            "reachability_limited": "调整站位、工具长度或工位布局，减少过伸操作。",
            "reachability_overextended": "复核工位距离和工具长度，减少长时间伸手操作。",
            "operation_unstable": "复核操作点定位、工具支撑和人员姿态，降低反复试探。",
            "pose_missing": "检查手部轨迹文件时间轴或 offset 设置，确认该片段是否有有效采样。",
            "operation_without_visibility": "优先改善视野或补光，确保关键操作点可见后继续操作。",
        }
        return QwenExplanation(
            summary=explanations.get(event.risk_type, "检测到人因工效风险。"),
            recommendation=recommendations.get(event.risk_type, "建议人工复核该片段。"),
            risk_type=event.risk_type,
            risk_level=event.risk_level,
            evidence_description="该结论来自本地视频规则和手部轨迹融合分析，未调用 Qwen2.5-VL。",
            needs_human_review=event.risk_level == "high",
        )

    def _hand_pose_window_metrics(self, window: list[tuple[FrameAnalysis, Any]]) -> HandPoseMetrics:
        if self.hand_pose_timeline is None:
            return HandPoseMetrics()
        return self.hand_pose_timeline.window_metrics(window[0][0].timestamp_sec, window[-1][0].timestamp_sec)

    def _task_window(
        self,
        task_state: TaskState,
        window: list[tuple[FrameAnalysis, Any]],
        hand_pose_metrics: HandPoseMetrics,
    ) -> dict[str, Any]:
        start = window[0][0].timestamp_sec
        end = window[-1][0].timestamp_sec
        return {
            "time_range": [self._format_time(start), self._format_time(end)],
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "task": asdict(task_state),
            "hand_pose": {
                "matched": hand_pose_metrics.hand_pose_matched,
                "stability_score": hand_pose_metrics.stability_score,
                "reach_distance_mm": hand_pose_metrics.reach_distance_mm,
                "nearest_event_type": hand_pose_metrics.nearest_event_type,
                "time_to_nearest_event_sec": hand_pose_metrics.time_to_nearest_event_sec,
            },
        }

    def _merge_task_timeline(self, windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not windows:
            return []
        timeline: list[dict[str, Any]] = []
        current = {
            "time_range": list(windows[0]["time_range"]),
            "start_sec": windows[0]["start_sec"],
            "end_sec": windows[0]["end_sec"],
            **windows[0]["task"],
            "window_count": 1,
            "hand_pose_windows": 1 if windows[0].get("hand_pose", {}).get("matched") else 0,
        }
        confidences = [float(windows[0]["task"]["confidence"])]
        for window in windows[1:]:
            task = window["task"]
            if task["task_id"] == current["task_id"] and task["task_phase"] == current["task_phase"]:
                current["time_range"][1] = window["time_range"][1]
                current["end_sec"] = window["end_sec"]
                current["window_count"] += 1
                current["hand_pose_windows"] += 1 if window.get("hand_pose", {}).get("matched") else 0
                confidences.append(float(task["confidence"]))
                current["confidence"] = round(mean(confidences), 3)
                current["matched_targets"] = sorted(set(current["matched_targets"]) | set(task["matched_targets"]))
            else:
                timeline.append(current)
                current = {
                    "time_range": list(window["time_range"]),
                    "start_sec": window["start_sec"],
                    "end_sec": window["end_sec"],
                    **task,
                    "window_count": 1,
                    "hand_pose_windows": 1 if window.get("hand_pose", {}).get("matched") else 0,
                }
                confidences = [float(task["confidence"])]
        timeline.append(current)
        return timeline

    def _task_coverage(self, windows: list[dict[str, Any]]) -> float:
        if not windows:
            return 0.0
        known = sum(1 for window in windows if window["task"]["task_id"] != UNKNOWN_TASK.task_id)
        return round(known / len(windows), 3)

    def _hand_pose_coverage(self, windows: list[dict[str, Any]]) -> float:
        if not windows:
            return 0.0
        matched = sum(1 for window in windows if window.get("hand_pose", {}).get("matched"))
        return round(matched / len(windows), 3)

    def _risk_level(self, metrics: ErgonomicMetrics, hand_pose_metrics: HandPoseMetrics) -> str:
        worst = min(metrics.visibility_score, metrics.reachability_score)
        if worst < 0.35 or metrics.occlusion_duration_sec >= 2.0:
            return "high"
        if hand_pose_metrics.hand_pose_matched:
            if hand_pose_metrics.stability_score is not None and hand_pose_metrics.stability_score < 0.25:
                return "high"
            if (hand_pose_metrics.reach_distance_mm or 0.0) >= 750.0:
                return "high"
        if worst < 0.6 or metrics.center_offset >= 0.72:
            return "medium"
        if hand_pose_metrics.hand_pose_matched:
            if hand_pose_metrics.stability_score is not None and hand_pose_metrics.stability_score < 0.45:
                return "medium"
            if (hand_pose_metrics.reach_distance_mm or 0.0) >= 650.0:
                return "medium"
        return "low"

    def _visibility_state(self, metrics: ErgonomicMetrics) -> str:
        if metrics.visibility_score < 0.35:
            return "occluded_or_unclear"
        if metrics.visibility_score < 0.65:
            return "partially_visible"
        return "visible"

    def _format_time(self, seconds: float) -> str:
        minutes, sec = divmod(seconds, 60)
        hours, minutes = divmod(int(minutes), 60)
        return f"{hours:02d}:{minutes:02d}:{sec:04.1f}"
