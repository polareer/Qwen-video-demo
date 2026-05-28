# PyTorch Image Recognition Project

This repository contains the original CIFAR-10/MNIST PyTorch examples and a first-person ergonomic video analysis MVP.

The ergonomic pipeline is designed for industrial operation videos:

```text
first-person video
-> frame sampling and window analysis
-> target / hand / image-quality perception
-> SOP task-template matching
-> visibility and reachability scoring
-> ergonomic risk events
-> keyframe extraction
-> Ollama + Qwen2.5-VL explanation
-> JSON report and evidence frames
```

## Setup

```bash
pip install -r requirements.txt
```

Optional enhancements:

- Install `ultralytics` and pass `--yolo-model` for YOLO/RT-DETR detection.
- Install `mediapipe` to improve hand landmark detection.
- Start Ollama and install `qwen2.5vl:3b` for local visual explanations.

## Run Ergonomic Analysis

Offline video with default task templates:

```bash
python analyze_ergonomics.py --source path/to/first_person_video.mp4
```

Disable Ollama and use local rule explanations only:

```bash
python analyze_ergonomics.py --source path/to/first_person_video.mp4 --no-ollama
```

Use a custom SOP task template:

```bash
python analyze_ergonomics.py --source path/to/video.mp4 --task-template configs/task_templates.yaml
```

Use a detector and preferred target label:

```bash
python analyze_ergonomics.py --source path/to/video.mp4 --yolo-model outputs/tool_detector.pt --target-label connector
```

## Outputs

The report is written to `outputs/ergonomics/*_ergonomic_report.json` by default and contains:

- `task_timeline`: recognized task phases, time ranges, confidence, and matched targets.
- `events`: risk events with task info, risk type, metrics, keyframes, and Qwen explanation.
- `summary`: event count, risk types, task coverage, high-risk count, and review flag.

Evidence frames are saved under `outputs/ergonomics/keyframes/`.

## Ollama

Confirm the local service and model:

```bash
ollama list
curl http://127.0.0.1:11434/api/tags
```

Default config:

```yaml
ollama:
  base_url: "http://127.0.0.1:11434"
  model: "qwen2.5vl:3b"
  timeout_seconds: 180
```

Qwen2.5-VL is used only after a rule-based risk event is triggered. It explains the saved keyframes and structured metrics; it does not calculate the risk scores or perform real-time detection.

## Original Image Classification Demo

Train the CIFAR-10 model:

```bash
python train.py
```
