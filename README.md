# Qwen Video Demo：第一人称人因工效视频分析

本项目用于跑通“前处理 + 小模型 + 手部轨迹 + 本地 Qwen 解释”的第一人称工业操作视频分析闭环。当前重点不是完整 SOP 合规判定，而是判断操作者在第一人称视角下的：

- 可视性：关键操作区域是否看得清、是否在视野中心、是否模糊或过暗。
- 可达性：手部相对头显/摄像头的位置、距离和运动趋势是否合理。
- 稳定性：手部轨迹是否抖动、过快、反复试探。
- 孔完成事件复核：孔完成时间来自手部轨迹 CSV/Excel 的事件记录，不依赖视觉检测铆钉孔。

核心链路：

```text
第一人称长视频
+ 手部轨迹 CSV/Excel
+ 手动时间偏移 offset
-> 帧采样与窗口切片
-> YOLO/OpenCV 前处理
-> 手部空间轨迹融合
-> 可视性 / 可达性 / 稳定性评分
-> 风险事件与孔完成窗口分析
-> 关键帧 + 结构化指标交给 Ollama/Qwen2.5-VL 解释
-> JSON 报告 + 证据帧
```

## 环境安装

基础依赖：

```bash
pip install -r requirements.txt
```

可选增强：

- 安装 `ultralytics` 后，可通过 `--yolo-model` 接入 YOLO / RT-DETR 小模型检测。
- 如果已有 CUDA 版 PyTorch，建议使用 `python -m pip install ultralytics --no-deps`，避免替换 GPU 版 PyTorch。
- 安装并启动 Ollama 后，可用本地 `qwen2.5vl:3b` 对风险关键帧做解释。

本机已验证：

```text
torch 2.5.1+cu121
cuda_available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

## 直接运行视频分析

纯视频分析：

```bash
python analyze_ergonomics.py --source path/to/video.mp4 --no-ollama
```

使用 YOLO 小模型走 GPU：

```bash
python analyze_ergonomics.py \
  --source path/to/video.mp4 \
  --yolo-model outputs/weights/yolo11n.pt \
  --device 0 \
  --detector-conf 0.25 \
  --detector-imgsz 640 \
  --no-ollama
```

融合手部轨迹 CSV：

```bash
python analyze_ergonomics.py \
  --source path/to/video.mp4 \
  --hand-pose-csv path/to/HandTrajectory_Right.csv \
  --hand-pose-offset-sec 0 \
  --no-ollama
```

同时使用 YOLO、GPU 和手部轨迹：

```bash
python analyze_ergonomics.py \
  --source outputs/vlm_test/hololens_trim.mp4 \
  --output-dir outputs/hole_event_test/report \
  --analysis-fps 5 \
  --yolo-model outputs/weights/yolo11n.pt \
  --device 0 \
  --hand-pose-csv "C:/Users/zzr/Downloads/HandTrajectory_Right_20260511_150317.csv" \
  --hand-pose-offset-sec 0 \
  --no-ollama
```

快速扫描视频与手部轨迹的时间偏移：

```bash
python tools/sweep_hand_pose_offset.py \
  --source outputs/vlm_test/hololens_trim.mp4 \
  --hand-pose-csv "C:/Users/zzr/Downloads/HandTrajectory_Right_20260511_150317.csv" \
  --start -5 \
  --end 5 \
  --step 0.5 \
  --output outputs/hole_event_test/offset_sweep.json
```

该工具只检查时间轴覆盖关系，不跑 YOLO 或 Qwen。它适合在换成完整视频后先判断 `--hand-pose-offset-sec` 的合理范围。

## 手部轨迹数据

当前支持 `#META + 轨迹行 + #EVENTS` 格式：

```text
#META ...
time,head_world_pos_x,...,center_x,center_y,center_z,Palm_x,...,ThumbTip_z
...
#EVENTS
event_type,hole_id,event_time
hole_completed,Ringline,5.9008
```

时间对齐关系：

```text
hand_pose_time = video_time + hand_pose_offset_sec
```

融合后会计算：

- `hand_pose_matched`：当前视频窗口是否匹配到手部轨迹。
- `center_relative_mm` / `palm_relative_mm`：手部相对头显/摄像头坐标。
- `hand_speed_mm_s`：窗口内手部速度。
- `stability_score`：手部稳定性评分。
- `reach_distance_mm`：手部相对头显距离。
- `posture_spread_mm`：手部关键点展开程度。
- `nearest_event_type` / `nearest_event_time`：最近孔完成事件。

## 孔完成事件分析

项目不会把“铆钉孔”作为视觉检测类别。孔完成节点直接来自 CSV/Excel 中的 `hole_completed` 事件。

每个孔完成事件会生成一个局部窗口：

```text
event_time - 2 秒  ->  event_time  ->  event_time + 1 秒
```

报告中的 `hole_event_analysis` 会包含：

- `hole_id`
- `event_time` 与对齐后 `video_time`
- 窗口内 `visibility_score`
- 窗口内 `reachability_score`
- 窗口内 `brightness`、`sharpness`
- 手部 `stability_score`、`reach_distance_mm`、`hand_speed_mm_s`
- `risk_level` 与 `risk_reasons`
- 事件前、中、后的关键帧证据

## 训练第一人称装配 YOLO 小模型

当前 YOLO 只建议训练三个粗粒度类别，用于增强前处理和可视性/可达性判断：

```text
0 tool
1 workpiece
2 hand
```

对应配置：

```text
configs/assembly_yolo.yaml
```

从视频抽帧供标注：

```bash
python tools/extract_yolo_frames.py \
  --source outputs/vlm_test/hololens_trim.mp4 \
  --output-dir datasets/assembly_yolo/images/unlabeled \
  --every-sec 0.5 \
  --max-frames 60 \
  --prefix hololens_assembly
```

标注后按 YOLO 格式放入：

```text
datasets/assembly_yolo/images/train
datasets/assembly_yolo/labels/train
datasets/assembly_yolo/images/val
datasets/assembly_yolo/labels/val
```

YOLO 标签格式：

```text
class_id x_center y_center width height
```

训练前检查：

```bash
python tools/train_yolo_assembly.py \
  --data configs/assembly_yolo.yaml \
  --model outputs/weights/yolo11n.pt \
  --device 0 \
  --dry-run
```

开始训练：

```bash
python tools/train_yolo_assembly.py \
  --data configs/assembly_yolo.yaml \
  --model outputs/weights/yolo11n.pt \
  --device 0 \
  --epochs 80 \
  --imgsz 640 \
  --batch 8
```

训练完成后接入分析：

```bash
python analyze_ergonomics.py \
  --source outputs/vlm_test/hololens_trim.mp4 \
  --yolo-model outputs/yolo_train/assembly_yolo11n/weights/best.pt \
  --device 0 \
  --target-label workpiece \
  --no-ollama
```

## Ollama / Qwen2.5-VL

确认本地 Ollama 和模型：

```bash
ollama list
curl http://127.0.0.1:11434/api/tags
```

默认配置：

```yaml
ollama:
  base_url: "http://127.0.0.1:11434"
  model: "qwen2.5vl:3b"
  timeout_seconds: 180
```

Qwen2.5-VL 只在风险事件触发后解释关键帧和结构化指标，不负责实时检测，也不负责计算风险分数。

## 输出结果

默认报告路径：

```text
outputs/ergonomics/*_ergonomic_report.json
```

主要字段：

- `task_timeline`：窗口级任务识别结果。
- `events`：风险事件列表，包含任务、风险类型、指标、关键帧和解释。
- `events[].hand_pose_metrics`：视频窗口融合的手部轨迹指标。
- `hole_event_analysis`：围绕每个 `hole_completed` 的前后窗口复核结果。
- `summary.hole_event_count`：CSV/Excel 中的孔完成事件总数。
- `summary.hole_event_analyzed_count`：落在视频时间轴内并完成复核的孔完成事件数。
- `summary.hole_event_skipped_count`：因视频片段不覆盖而跳过的孔完成事件数。
- `summary`：风险数量、风险类型、手部轨迹覆盖率、孔完成事件数等。

关键帧默认保存到：

```text
outputs/ergonomics/keyframes/
```

## 原始图像分类示例

仓库仍保留原始 PyTorch 图像分类示例：

```bash
python train.py
```
