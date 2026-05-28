import argparse

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_PROMPT = (
    "请详细描述这张图，并重点说明："
    "1. 画面中的关键对象；"
    "2. 如果这是装配或操作场景，正在执行的动作；"
    "3. 可能的风险点或易错点。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use Qwen2.5-VL to understand an image.")
    parser.add_argument("--image", required=True, help="Path to the image file.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Question for the model.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model name or local path.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum output tokens.")
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


def build_messages(image_path: str, prompt: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def main() -> None:
    args = parse_args()
    model, processor = load_model(args.model_name)
    messages = build_messages(args.image, args.prompt)

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

    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text[0])


if __name__ == "__main__":
    main()
