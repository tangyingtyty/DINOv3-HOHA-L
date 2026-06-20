# Exploring Vision Foundation Models with High-Order Hypergraph Adaptation for High-Resolution Remote Sensing Image Semantic Segmentation

This repository implements a semantic segmentation framework for remote-sensing imagery. The core idea is to **freeze a Vision Foundation Model (VFM) backbone** and attach a lightweight **HyperGraph Adapter** together with a UPerNet-style decode head, so that general-purpose visual representations can be transferred to remote-sensing segmentation with very few trainable parameters.

This uploaded version provides a complete implementation using the **DINOv3** backbone. By design, the framework also supports DINOv2, SAM / SAM3, MAE, and Openclip.

---

## Requirements

Main dependencies (Python ≥ 3.10):

```bash
pip install torch torchvision
pip install opencv-python albumentations numpy tqdm
pip install segmentation-models-pytorch ttach
```

You also need the **official DINOv3 repository and pretrained weights** locally, and must update the paths at the top of `model/dinov3_hypergraph_upernet.py`:

```python
DINOV3_REPO_DIR  = '/path/to/dinov3'                    # Local path to the DINOv3 code repo
DINOV3_CHECKPOINTS = DINOV3_REPO_DIR + '/checkpoints/'  # Directory of pretrained weights
```

The supported DINOv3 backbones and their weight filenames are listed in the `DINOV3_WEIGHTS` dictionary (ViT / ConvNeXt / satellite-pretrained SAT, etc.).

---

## Data Preparation

### Dataset Acquisition

- **UAVid**, **LoveDA**, and **Potsdam** are public benchmarks and can be obtained freely from their official sources:
  - UAVid: <https://uavid.nl/>
  - LoveDA: <https://github.com/Junjue-Wang/LoveDA>
  - Potsdam (ISPRS 2D Semantic Labeling): <https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx>
- The **Anhui** and **Hainan** datasets are proprietary production data that are confidential and **cannot be made publicly available**.

### Directory Layout

Datasets are organized under `data/<dataset_name>/` and split into `train` / `test`, each containing `image` / `label` subdirectories. Labels are single-channel grayscale images (pixel value = class index), with matching filenames:

```
data/<dataset_name>/
├── train/
│   ├── image/        # Input images (RGB)
│   │   ├── 0001.png
│   │   └── ...
│   └── label/        # Corresponding labels (grayscale, pixel value = class id)
│       ├── 0001.png
│       └── ...
└── test/
    ├── image/        # Input images (RGB)
    │   ├── 0001.png
    │   └── ...
    └── label/        # Corresponding labels (grayscale, pixel value = class id)
        ├── 0001.png
        └── ...
```

Pass these to training with `--train_dir data/<dataset_name>/train` and `--test_dir data/<dataset_name>/test`.

---

## Training

```bash
python train.py \
    --encoder dinov3_vitl16 \
    --architecture HyperGraphUPerNet \
    --train_dir data/your_dataset/train \
    --test_dir  data/your_dataset/test \
    --batch_size 4 \
    --num_epochs 80 \
    --learning_rate 3e-4 \
    --optimizer adam \
    --scheduler cos
```

Common arguments (see `train.py`):

| Argument | Description | Default |
|----------|-------------|---------|
| `--encoder` | Backbone name | `dinov3_vitl16` |
| `--architecture` | Decode architecture (fixed to HyperGraph UPerNet for the DINOv3 backbone) | `HyperGraphUPerNet` |
| `--pretrained` | Checkpoint directory name for resuming | `None` |

---

## Evaluation Metrics

`metric.py` computes the following from the confusion matrix:

- Overall Accuracy **Acc**
- mean Intersection-over-Union **mIoU**
- mean **F1** (mF1)
- Per-class IoU / F1 / Precision / Recall

---

## Citation

This code is the experimental implementation for the paper (Paper 2). If this repository is helpful to your research, please cite the corresponding paper and acknowledge the following foundational works: DINOv3, UPerNet, `segmentation_models_pytorch`, and `ttach`.
