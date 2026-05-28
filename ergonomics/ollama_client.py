"""Ollama/Qwen2.5-VL client for visual ergonomic explanations."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from .schemas import ErgonomicRiskEvent, QwenExplanation


SYSTEM_PROMPT = """你是工业人因工效分析助手。请基于第一人称视角关键帧和结构化检测结果，
判断当前操作是否存在可视性、可达性或可操作性风险。只基于图像证据和输入数据解释，
不要臆测看不见的信息。

请严格输出 JSON，字段为：
{
  "risk_type": "风险类型",
  "risk_level": "低/中/高",
  "evidence_description": "证据描述",
  "summary": "原因解释",
  "recommendation": "改善建议",
  "needs_human_review": true 或 false
}
"""


class OllamaVisionClient:
    """Small wrapper around Ollama's local chat API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5vl:3b",
        timeout_seconds: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        try:
            self.list_models()
            return True
        except OSError:
            return False

    def list_models(self) -> list[str]:
        url = f"{self.base_url}/api/tags"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
        return [item.get("name", "") for item in data.get("models", [])]

    def explain_event(self, event: ErgonomicRiskEvent) -> QwenExplanation:
        images = [self._encode_image(path) for path in event.evidence.keyframes[:4] if Path(path).exists()]
        user_prompt = (
            "请分析以下第一人称人因工效风险事件。\n"
            "结构化事件数据如下：\n"
            f"{json.dumps(event.to_dict(), ensure_ascii=False, indent=2)}\n"
            "请只基于这些数据和关键帧给出解释，重点说明任务、可视性、可达性和是否需要人工复核。"
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt, "images": images},
            ],
            "format": "json",
            "options": {"temperature": 0.1, "num_ctx": 512},
        }
        try:
            response = self._post_json("/api/chat", payload)
            content = response.get("message", {}).get("content", "")
            return self._parse_explanation(content)
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return QwenExplanation(
                summary="Ollama/Qwen2.5-VL 暂不可用，已保留规则引擎风险结果。",
                recommendation="请确认 Ollama 服务已启动、模型名配置正确，并在资源充足时重新生成解释。",
                raw_response=str(exc),
                needs_human_review=event.risk_level == "high",
            )

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _parse_explanation(self, content: str) -> QwenExplanation:
        parsed = json.loads(content)
        return QwenExplanation(
            summary=str(parsed.get("summary", "")),
            recommendation=str(parsed.get("recommendation", "")),
            risk_type=parsed.get("risk_type"),
            risk_level=parsed.get("risk_level"),
            evidence_description=parsed.get("evidence_description"),
            needs_human_review=parsed.get("needs_human_review"),
            raw_response=content,
        )

    def _encode_image(self, image_path: str, max_size: int = 512) -> str:
        path = Path(image_path)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((max_size, max_size))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
