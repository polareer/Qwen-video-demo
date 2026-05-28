import argparse
import json
import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import threading

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model state
model = None
processor = None
model_lock = threading.Lock()
model_loaded = False

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


def load_model_once(model_name):
    global model, processor, model_loaded
    with model_lock:
        if model_loaded:
            return
        print(f"Loading model {model_name}...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(model_name)
        model_loaded = True
        print("Model loaded successfully!")


@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')


@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        'model_loaded': model_loaded,
        'model_name': DEFAULT_MODEL_NAME
    })


@app.route('/api/load-model', methods=['POST'])
def api_load_model():
    data = request.json or {}
    model_name = data.get('model_name', DEFAULT_MODEL_NAME)
    try:
        thread = threading.Thread(target=load_model_once, args=(model_name,))
        thread.start()
        return jsonify({'status': 'loading'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded. Please load the model first.'}), 400

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    prompt = request.form.get('prompt', DEFAULT_PROMPT)
    fps = float(request.form.get('fps', 1.0))
    max_pixels = int(request.form.get('max_pixels', 360 * 420))
    max_new_tokens = int(request.form.get('max_new_tokens', 1024))

    # Save uploaded video
    filename = f"{uuid.uuid4()}_{video_file.filename}"
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(video_path)

    try:
        # Build messages
        messages = [
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

        # Generate response
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
        )[0]

        # Try to parse as JSON
        result = {'raw_text': output_text}
        try:
            parsed = json.loads(output_text)
            result['parsed'] = parsed
        except json.JSONDecodeError:
            pass

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(video_path):
            os.remove(video_path)


def main():
    parser = argparse.ArgumentParser(description="AR Assembly Video Analysis Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    parser.add_argument('--preload-model', action='store_true', help='Preload the model on startup')
    args = parser.parse_args()

    if args.preload_model:
        load_model_once(DEFAULT_MODEL_NAME)

    print(f"Starting server on {args.host}:{args.port}")
    print(f"Access the frontend at: http://localhost:{args.port}/")
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
