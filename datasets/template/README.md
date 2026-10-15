# Dataset template

Use this layout for each dataset:

- `data.yaml`: main YOLO11 training configuration
- `images/train/` and `labels/train/`: main training samples
- `images/val/` and `labels/val/`: main validation samples
- `images/unlabeled/`: images waiting for automatic labeling
- `labels/unlabeled/`: generated and manually corrected labels
- `label_model/`: the small manually labeled dataset used to train the auto-label model
- `label_model/data.yaml`: independent configuration for the auto-label model

Image files and label files use the same relative path and filename stem. YOLO label files contain normalized `class_id center_x center_y width height` values. Class IDs start at `0` and match the order in `names`.
