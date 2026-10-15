# Dataset template

Use this layout for a dataset:

- `images/train/` and `labels/train/`: training samples
- `images/val/` and `labels/val/`: validation samples
- `images/unlabeled/`: images waiting for automatic labeling
- `labels/unlabeled/`: generated and manually corrected labels
- `label_model/`: the small manually labeled dataset used to train the auto-label model
