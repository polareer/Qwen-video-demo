import argparse
import json

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_PROMPT = """请理解这个第一人称增强现实装配视频，并输出一个结构化分析结果。

请尽量按 JSON 返回，字段建议如下：
{
  "video_summary": "一句话概述视频内容",
  "assembly_goal": "本次装配任务目标",
  "steps": [
    {
      "step_id": 1,
      "time_range": "起止时间，未知可写 approximate",
      "action": "执行了什么动作",
      "objects": ["零件/工具/设备"],
      "ar_guidance": "画面中的 AR 指引信息",
      "possible_issue": "可能的问题、模糊点或风险",
      "confidence": "high/medium/low"
    }
  ],
  "overall_risks": ["整体风险点"],
  "improvement_suggestions": [
    "如何改进视频理解系统",
    "如何改进装配流程或 AR 引导"
  ]
}

如果无法精确定位时间，请明确说明是近似判断，但仍然要尽量拆出步骤。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Qwen2.5-VL to understand a first-person assembly video."
    )
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Question for the model.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model name or local path.")
    parser.add_argument("--fps", type=float, default=1.0, help="Video sampling FPS.")
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=360 * 420,
        help="Upper bound of pixels per frame fed to the processor.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum output tokens.")
    parser.add_argument(
        "--pretty-json",
        action="store_true",
        help="Pretty print the output if the model returns valid JSON.",
    )
    return parser.parse_args()


def load_model(model_name: str):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def build_messages(video_path: str, prompt: str, fps: float, max_pixels: int):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def generate_response(model, processor, messages, max_new_tokens: int) -> str:
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def main() -> None:
    args = parse_args()
    model, processor = load_model(args.model_name)
    messages = build_messages(args.video, args.prompt, args.fps, args.max_pixels)
    result = generate_response(model, processor, messages, args.max_new_tokens)

    if args.pretty_json:
        try:
            parsed = json.loads(result)
            print(json.dumps(parsed, ensure_ascii=False, indent=2))
            return
        except json.JSONDecodeError:
            pass

    print(result)


if __name__ == "__main__":
    main()
