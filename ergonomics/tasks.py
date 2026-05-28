"""SOP task templates and window-level task recognition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from .schemas import TaskState
from .vision import FrameAnalysis


UNKNOWN_TASK = TaskState(
    task_id="unknown_task",
    task_name="未知任务",
    task_phase="unknown",
    confidence=0.0,
    matched_targets=[],
    evidence="未匹配到足够的目标或手部关系证据。",
)


@dataclass
class TaskTemplate:
    task_id: str
    name: str
    phase: str = "operation"
    expected_targets: list[str] = field(default_factory=list)
    expected_hand_relation: str = "hand_near_target"
    min_duration_sec: float = 1.0
    visibility_threshold: float | None = None
    reachability_threshold: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskTemplate":
        return cls(
            task_id=str(data.get("id", data.get("task_id", "unknown_task"))),
            name=str(data.get("name", data.get("id", "未知任务"))),
            phase=str(data.get("phase", data.get("task_phase", "operation"))),
            expected_targets=[str(item) for item in data.get("expected_targets", [])],
            expected_hand_relation=str(data.get("expected_hand_relation", "hand_near_target")),
            min_duration_sec=float(data.get("min_duration_sec", 1.0)),
            visibility_threshold=_optional_float(data.get("visibility_threshold")),
            reachability_threshold=_optional_float(data.get("reachability_threshold")),
        )


class TaskRecognizer:
    """Matches short first-person video windows against configured SOP templates."""

    def __init__(self, templates: list[TaskTemplate], unknown_threshold: float = 0.38) -> None:
        self.templates = templates
        self.unknown_threshold = unknown_threshold

    @classmethod
    def from_yaml(cls, path: str | None) -> "TaskRecognizer":
        if not path:
            return cls([])
        template_path = Path(path)
        if not template_path.exists():
            raise FileNotFoundError(f"Task template file not found: {path}")
        data = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
        templates = [TaskTemplate.from_dict(item) for item in data.get("tasks", [])]
        return cls(templates, unknown_threshold=float(data.get("unknown_threshold", 0.38)))

    def recognize(self, analyses: list[FrameAnalysis]) -> TaskState:
        if not self.templates or not analyses:
            return UNKNOWN_TASK

        labels = self._window_labels(analyses)
        best_state = UNKNOWN_TASK
        best_score = 0.0
        for template in self.templates:
            state = self._score_template(template, analyses, labels)
            if state.confidence > best_score:
                best_state = state
                best_score = state.confidence
        if best_score < self.unknown_threshold:
            return UNKNOWN_TASK
        return best_state

    def _score_template(
        self,
        template: TaskTemplate,
        analyses: list[FrameAnalysis],
        labels: list[str],
    ) -> TaskState:
        matched_targets = self._matched_targets(template.expected_targets, labels)
        if template.expected_targets:
            target_score = len(matched_targets) / len(template.expected_targets)
        else:
            target_score = 0.25

        visibility = mean(item.visibility_score for item in analyses)
        reachability = mean(item.reachability_score for item in analyses)
        relation_score = self._relation_score(template.expected_hand_relation, analyses)
        threshold_bonus = self._threshold_bonus(template, visibility, reachability)
        confidence = (target_score * 0.45) + (relation_score * 0.30) + (visibility * 0.15) + threshold_bonus
        confidence = round(max(0.0, min(confidence, 1.0)), 3)

        evidence = (
            f"matched_targets={matched_targets or []}; "
            f"relation={template.expected_hand_relation}; "
            f"visibility={visibility:.2f}; reachability={reachability:.2f}"
        )
        return TaskState(
            task_id=template.task_id,
            task_name=template.name,
            task_phase=template.phase,
            confidence=confidence,
            matched_targets=matched_targets,
            evidence=evidence,
        )

    def _window_labels(self, analyses: list[FrameAnalysis]) -> list[str]:
        labels: list[str] = []
        for analysis in analyses:
            labels.append(analysis.target.label.lower())
            labels.extend(det.label.lower() for det in analysis.hands)
            labels.extend(det.label.lower() for det in analysis.detections)
        return sorted(set(labels))

    def _matched_targets(self, expected_targets: list[str], labels: list[str]) -> list[str]:
        matched: list[str] = []
        for expected in expected_targets:
            expected_lower = expected.lower()
            if any(expected_lower in label or label in expected_lower for label in labels):
                matched.append(expected)
        return matched

    def _relation_score(self, relation: str, analyses: list[FrameAnalysis]) -> float:
        reachability = mean(item.reachability_score for item in analyses)
        has_hand_ratio = sum(1 for item in analyses if item.hands) / len(analyses)
        occlusion = mean(item.occlusion_ratio for item in analyses)
        relation_lower = relation.lower()
        if relation_lower in {"hand_near_target", "tool_near_target"}:
            return float(min((reachability * 0.75) + (has_hand_ratio * 0.25), 1.0))
        if relation_lower in {"target_visible", "inspect_target"}:
            return mean(item.visibility_score for item in analyses)
        if relation_lower in {"contact_or_occlusion", "hand_over_target"}:
            return float(min((occlusion * 0.7) + (has_hand_ratio * 0.3), 1.0))
        return 0.5

    def _threshold_bonus(self, template: TaskTemplate, visibility: float, reachability: float) -> float:
        bonus = 0.0
        if template.visibility_threshold is not None and visibility >= template.visibility_threshold:
            bonus += 0.05
        if template.reachability_threshold is not None and reachability >= template.reachability_threshold:
            bonus += 0.05
        return bonus


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
