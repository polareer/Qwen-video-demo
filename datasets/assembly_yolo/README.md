# Assembly YOLO Dataset

This directory is intentionally ignored by git because it can contain large videos, extracted frames, and labels.

Class ids:

```text
0 tool
1 workpiece
2 hand
```

Workflow:

1. Review images in `images/unlabeled`.
2. Annotate boxes with LabelImg, CVAT, Roboflow, or any YOLO-compatible tool.
3. Put training images in `images/train` and labels in `labels/train`.
4. Put validation images in `images/val` and labels in `labels/val`.
5. Train:

```powershell
python tools\train_yolo_assembly.py --data configs\assembly_yolo.yaml --model outputs\weights\yolo11n.pt --device 0
```

YOLO label format per line:

```text
class_id x_center y_center width height
```

All coordinates are normalized to 0-1.
