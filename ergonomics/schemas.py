"""Typed data structures for ergonomic risk analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: list[int]
    source: str = "heuristic"


@dataclass
class TaskState:
    task_id: str
    task_name: str
    task_phase: str
    confidence: float
    matched_targets: list[str] = field(default_factory=list)
    evidence: str = ""


@dataclass
class TargetState:
    name: str
    visibility: str
    bbox: list[int] | None = None


@dataclass
class ErgonomicMetrics:
    visibility_score: float
    reachability_score: float
    occlusion_duration_sec: float
    center_offset: float
    brightness: float
    sharpness: float


@dataclass
class HandPoseMetrics:
    hand_pose_matched: bool = False
    center_relative_mm: list[float] | None = None
    palm_relative_mm: list[float] | None = None
    hand_speed_mm_s: float | None = None
    stability_score: float | None = None
    reach_distance_mm: float | None = None
    posture_spread_mm: float | None = None
    nearest_event_type: str | None = None
    nearest_event_time: float | None = None
    time_to_nearest_event_sec: float | None = None
    sample_count: int = 0
    source_time_range: list[float] | None = None


@dataclass
class Evidence:
    keyframes: list[str] = field(default_factory=list)
    clip: str | None = None


@dataclass
class QwenExplanation:
    summary: str = ""
    recommendation: str = ""
    risk_type: str | None = None
    risk_level: str | None = None
    evidence_description: str | None = None
    needs_human_review: bool | None = None
    raw_response: str | None = None


@dataclass
class ErgonomicRiskEvent:
    event_id: str
    video_id: str
    time_range: list[str]
    risk_type: str
    risk_level: str
    task_id: str
    task_name: str
    task_confidence: float
    task_phase: str
    target: TargetState
    metrics: ErgonomicMetrics
    evidence: Evidence
    hand_pose_metrics: HandPoseMetrics = field(default_factory=HandPoseMetrics)
    qwen_explanation: QwenExplanation = field(default_factory=QwenExplanation)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
