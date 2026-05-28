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
    qwen_explanation: QwenExplanation = field(default_factory=QwenExplanation)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
