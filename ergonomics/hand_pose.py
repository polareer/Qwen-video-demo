"""Hand trajectory parsing and window-level pose metrics."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .schemas import HandPoseMetrics


JOINT_NAMES = ("Palm", "Wrist", "IndexTip", "MiddleTip", "ThumbTip")


@dataclass
class HandPoseEvent:
    event_type: str
    hole_id: str
    event_time: float


@dataclass
class HandPoseSample:
    time: float
    head_world_pos: list[float]
    head_world_quat_xyzw: list[float]
    head_forward: list[float]
    center: list[float]
    valid_joint_count: int
    joints: dict[str, list[float]]


class HandPoseTimeline:
    """Time-indexed hand pose stream exported from the AR capture tool."""

    def __init__(
        self,
        samples: list[HandPoseSample],
        events: list[HandPoseEvent] | None = None,
        meta: dict[str, str] | None = None,
        offset_sec: float = 0.0,
        max_nearest_gap_sec: float = 0.8,
    ) -> None:
        self.samples = sorted(samples, key=lambda item: item.time)
        self.events = sorted(events or [], key=lambda item: item.event_time)
        self.meta = meta or {}
        self.offset_sec = offset_sec
        self.max_nearest_gap_sec = max_nearest_gap_sec

    @classmethod
    def from_csv(cls, path: str, offset_sec: float = 0.0) -> "HandPoseTimeline":
        csv_path = Path(path)
        if csv_path.suffix.lower() != ".csv":
            raise ValueError("Only CSV hand pose files are supported in this MVP. Export Excel files as CSV first.")
        meta: dict[str, str] = {}
        samples: list[HandPoseSample] = []
        events: list[HandPoseEvent] = []
        mode = "meta"
        header: list[str] | None = None
        event_header: list[str] | None = None

        for raw_line in csv_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#META"):
                key_value = line.replace("#META", "", 1).strip()
                if "=" in key_value:
                    key, value = key_value.split("=", 1)
                    meta[key] = value
                continue
            if line == "#EVENTS":
                mode = "events_header"
                continue
            if mode == "meta":
                header = next(csv.reader([line]))
                mode = "samples"
                continue
            if mode == "samples":
                if header is None:
                    continue
                row = next(csv.reader([line]))
                if len(row) != len(header):
                    continue
                row_data = dict(zip(header, row))
                sample = cls._parse_sample(row_data)
                if sample is not None:
                    samples.append(sample)
                continue
            if mode == "events_header":
                event_header = next(csv.reader([line]))
                mode = "events"
                continue
            if mode == "events" and event_header is not None:
                row = next(csv.reader([line]))
                event = cls._parse_event(dict(zip(event_header, row)))
                if event is not None:
                    events.append(event)

        return cls(samples=samples, events=events, meta=meta, offset_sec=offset_sec)

    def window_metrics(self, video_start_sec: float, video_end_sec: float) -> HandPoseMetrics:
        pose_start = video_start_sec + self.offset_sec
        pose_end = video_end_sec + self.offset_sec
        nearest_event = self._nearest_event((pose_start + pose_end) / 2)
        samples = [sample for sample in self.samples if pose_start <= sample.time <= pose_end]

        if not samples:
            nearest = self._nearest_sample((pose_start + pose_end) / 2)
            if nearest is not None and abs(nearest.time - ((pose_start + pose_end) / 2)) <= self.max_nearest_gap_sec:
                samples = [nearest]

        if not samples:
            return HandPoseMetrics(
                hand_pose_matched=False,
                nearest_event_type=nearest_event.event_type if nearest_event else None,
                nearest_event_time=nearest_event.event_time if nearest_event else None,
                time_to_nearest_event_sec=self._event_delta(nearest_event, (pose_start + pose_end) / 2),
                sample_count=0,
                source_time_range=[round(pose_start, 3), round(pose_end, 3)],
            )

        center = self._mean_point([sample.center for sample in samples])
        palm = self._mean_point([sample.joints["Palm"] for sample in samples if "Palm" in sample.joints])
        speeds = self._center_speeds(samples)
        speed = round(mean(speeds), 3) if speeds else 0.0
        stability_score = self._stability_score(samples, speeds)
        reach_distance = round(math.dist(center, [0.0, 0.0, 0.0]), 3)
        posture_spread = self._posture_spread(samples)

        return HandPoseMetrics(
            hand_pose_matched=True,
            center_relative_mm=[round(value, 3) for value in center],
            palm_relative_mm=[round(value, 3) for value in palm] if palm else None,
            hand_speed_mm_s=speed,
            stability_score=stability_score,
            reach_distance_mm=reach_distance,
            posture_spread_mm=posture_spread,
            nearest_event_type=nearest_event.event_type if nearest_event else None,
            nearest_event_time=nearest_event.event_time if nearest_event else None,
            time_to_nearest_event_sec=self._event_delta(nearest_event, (pose_start + pose_end) / 2),
            sample_count=len(samples),
            source_time_range=[round(samples[0].time, 3), round(samples[-1].time, 3)],
        )

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @staticmethod
    def _parse_sample(row: dict[str, str]) -> HandPoseSample | None:
        try:
            time = float(row["time"])
            joints = {
                name: [float(row[f"{name}_x"]), float(row[f"{name}_y"]), float(row[f"{name}_z"])]
                for name in JOINT_NAMES
                if row.get(f"{name}_x") not in {None, ""}
            }
            return HandPoseSample(
                time=time,
                head_world_pos=[
                    float(row["head_world_pos_x"]),
                    float(row["head_world_pos_y"]),
                    float(row["head_world_pos_z"]),
                ],
                head_world_quat_xyzw=[
                    float(row["head_world_qx"]),
                    float(row["head_world_qy"]),
                    float(row["head_world_qz"]),
                    float(row["head_world_qw"]),
                ],
                head_forward=[
                    float(row["head_forward_x"]),
                    float(row["head_forward_y"]),
                    float(row["head_forward_z"]),
                ],
                center=[float(row["center_x"]), float(row["center_y"]), float(row["center_z"])],
                valid_joint_count=int(float(row.get("valid_joint_count", 0))),
                joints=joints,
            )
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _parse_event(row: dict[str, str]) -> HandPoseEvent | None:
        try:
            return HandPoseEvent(
                event_type=str(row["event_type"]),
                hole_id=str(row.get("hole_id", "")),
                event_time=float(row["event_time"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _nearest_sample(self, time_sec: float) -> HandPoseSample | None:
        if not self.samples:
            return None
        return min(self.samples, key=lambda item: abs(item.time - time_sec))

    def _nearest_event(self, time_sec: float) -> HandPoseEvent | None:
        if not self.events:
            return None
        return min(self.events, key=lambda item: abs(item.event_time - time_sec))

    def _event_delta(self, event: HandPoseEvent | None, time_sec: float) -> float | None:
        if event is None:
            return None
        return round(event.event_time - time_sec, 3)

    def _center_speeds(self, samples: list[HandPoseSample]) -> list[float]:
        speeds: list[float] = []
        for previous, current in zip(samples, samples[1:]):
            delta_t = current.time - previous.time
            if delta_t > 0:
                speeds.append(math.dist(previous.center, current.center) / delta_t)
        return speeds

    def _stability_score(self, samples: list[HandPoseSample], speeds: list[float]) -> float:
        if len(samples) <= 1:
            return 0.75
        centers = [sample.center for sample in samples]
        axis_deviation = sum(pstdev([point[index] for point in centers]) for index in range(3))
        speed_penalty = min((mean(speeds) if speeds else 0.0) / 350.0, 1.0)
        jitter_penalty = min(axis_deviation / 160.0, 1.0)
        return round(max(0.0, 1.0 - (speed_penalty * 0.55) - (jitter_penalty * 0.45)), 3)

    def _posture_spread(self, samples: list[HandPoseSample]) -> float | None:
        spreads: list[float] = []
        for sample in samples:
            palm = sample.joints.get("Palm")
            if palm is None:
                continue
            distances = [
                math.dist(palm, joint)
                for name, joint in sample.joints.items()
                if name != "Palm"
            ]
            if distances:
                spreads.append(max(distances))
        if not spreads:
            return None
        return round(mean(spreads), 3)

    def _mean_point(self, points: list[list[float]]) -> list[float]:
        if not points:
            return []
        return [mean(point[index] for point in points) for index in range(3)]


def load_hand_pose_timeline(path: str | None, offset_sec: float = 0.0) -> HandPoseTimeline | None:
    if not path:
        return None
    return HandPoseTimeline.from_csv(path, offset_sec=offset_sec)
