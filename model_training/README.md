# Model Training — Infrastructure Defect Segmentation

This folder contains the complete training pipeline for the SegFormer-B2 model.

## 📁 Files

| File | Purpose |
|------|---------|
| **train.py** | Main training loop (2-stage: warmup + fine-tuning) with checkpointing & resume support |
| **dataset.py** | CODEBRIM dataset class with VOC-XML support and augmentation pipeline |
| **model.py** | SegFormer-B2 architecture, multi-label loss functions, and early stopping |
| **evaluate.py** | Evaluation metrics (mIoU, F1, per-class IoU) & visualization |
| **eval_output/** | Evaluation visualizations and detailed results |

## 🚀 Quick Start

### Option 1: Default Settings (Recommended)

From the project root:

```bash
cd model_training
python train.py
```

This will use defaults:
- ✅ Load dataset from `../roboflow_data/`
- ✅ Save checkpoints to `../checkpoints/`
- ✅ Train for 50 epochs (5 warmup + 45 main)
- ✅ Batch size: 4

### Option 2: Custom Settings

From `model_training/` directory:

```bash
python train.py \
  --data_dir ../roboflow_data \
  --out_dir ../checkpoints \
  --batch_size 8 \
  --warmup_epochs 5 \
  --main_epochs 45
```

### Option 3: Resume Training

```bash
python train.py \
  --data_dir ../roboflow_data \
  --out_dir ../checkpoints \
  --resume ../checkpoints/last.pth
```

## 📊 Training Pipeline

### Stage 1: Decoder Warmup (5 epochs)

**Purpose:** Adapt the pretrained ImageNet encoder to defect segmentation

- **Encoder:** Frozen (uses ImageNet features)
- **Decoder:** Learning rate 6e-4 with cosine decay
- **Loss:** Cross-entropy + Dice Loss (0.5 weight each)
- **Output:** `../checkpoints/stage1_final.pth`
- **Benefit:** Prevents catastrophic forgetting of pretrained weights

### Stage 2: Full Fine-tuning (45 epochs)

**Purpose:** End-to-end optimization for defect detection

- **Encoder:** Unfrozen with layer-wise learning rate decay (LR = 6e-5)
- **Decoder:** Learning rate 6e-4 (continued from warmup)
- **Loss Composition:**
  - Cross-Entropy: 0.4 weight (base classification loss)
  - Dice Loss: 0.4 weight (handles class imbalance)
  - Boundary Loss: 0.2 weight (sharpens defect edges)
- **Optimization:** Mixed precision (FP16) for faster training
- **Regularization:** Gradient clipping (max_norm=1.0)
- **Early Stopping:** Stop on val mIoU plateau (patience=10)
- **Output:** `../checkpoints/best_model.pth` (best validation performance)

## 🎯 Input & Output

### Input Format

VOC-XML annotations with bounding boxes:

```
roboflow_data/
├── image_0000005.jpg
├── image_0000005.xml  ← Bounding box annotations
├── image_0000021.jpg
├── image_0000021.xml
└── ... (more image/xml pairs)
```

XML structure:

```xml
<annotation>
  <filename>image_0000005.jpg</filename>
  <size>
    <width>1920</width>
    <height>1080</height>
  </size>
  <object>
    <name>crack</name>  <!-- one of: crack, spalling, corrosion, deformation -->
    <bndbox>
      <xmin>100</xmin>
      <ymin>200</ymin>
      <xmax>400</xmax>
      <ymax>500</ymax>
    </bndbox>
  </object>
</annotation>
```

### Output Files

After training completes:

```
../checkpoints/
├── best_model.pth       ← Best model (use for inference)
├── last.pth             ← Latest checkpoint (for resuming)
├── stage1_final.pth     ← Warmup stage completion
├── train_log.json       ← Training metrics per epoch
└── test_results.json    ← Final evaluation metrics
```

## 📈 Augmentation Pipeline

Applied during training for robustness:

| Augmentation | Probability | Purpose |
|---|---|---|
| **Horizontal Flip** | 50% | Symmetry robustness |
| **Rotate 90°** | 30% | Orientation variance |
| **Elastic Transform** | 20% | Irregular crack shapes |
| **Grid Distortion** | 20% | Non-linear deformations |
| **Color Jitter** | 40% | Lighting conditions |
| **CutMix** | 30% | Occlusion robustness |

## 🔧 Hyperparameters

Key settings (from `train.py`):

| Parameter | Value | Notes |
|---|---|---|
| **Input Resolution** | 512×512 | Padding preserves aspect ratio |
| **Batch Size (Stage 1)** | 4 | Decoder warmup |
| **Batch Size (Stage 2)** | 4 | Full fine-tuning |
| **Warmup Epochs** | 5 | Decoder-only training |
| **Main Epochs** | 45 | Full model fine-tuning |
| **Encoder LR (Stage 2)** | 6e-5 | Conservative learning rate |
| **Decoder LR (Stage 2)** | 6e-4 | Faster adaptation |
| **Warmup LR Schedule** | Linear → Cosine | Stable training start |
| **Gradient Clipping** | 1.0 | Prevents divergence |
| **Sigmoid Threshold** | 0.45 | Binary classification cutoff |
| **Early Stopping Patience** | 10 | Stop if no val mIoU improvement |

## 📊 Expected Performance

On CODEBRIM dataset (after full training):

| Metric | Expected Value | Notes |
|---|---|---|
| **Overall mIoU** | 0.48–0.55 | Depends on data quality |
| **Crack IoU** | 0.60–0.68 | Largest class, well-represented |
| **Spalling IoU** | 0.40–0.50 | Moderate frequency |
| **Corrosion IoU** | 0.35–0.45 | Rare, challenging |
| **Deformation IoU** | 0.30–0.40 | Rarest class |
| **Training Time (GPU)** | ~3 hours | NVIDIA T4 or equivalent |
| **Inference Latency** | ~32ms | Per image, single GPU |

## 🐛 Troubleshooting

### CUDA Out of Memory

Reduce batch size:

```bash
python train.py --batch_size 2
```

### Training Loss Not Decreasing

Check:
1. Dataset is correctly loaded: verify images & XMLs are readable
2. Learning rate is appropriate (try starting with 1e-3)
3. Number of training samples is sufficient (minimum ~500)

### Model Checkpoints Not Saving

Ensure `../checkpoints/` directory exists and is writable:

```bash
mkdir -p ../checkpoints
```

### Resume Training Fails

Check checkpoint file exists:

```bash
ls ../checkpoints/last.pth
```

## 📚 Key Concepts

### Two-Stage Training

**Why?** Fine-tuning pretrained models from random initializations risks catastrophic forgetting of ImageNet features. Stage 1 warmup lets the decoder adapt first, then Stage 2 unfreezes the encoder.

### Composite Loss Function

**Why?** Single Cross-Entropy loss biases toward the dominant class (background + cracks ~80% of pixels). Dice loss handles imbalance, and boundary loss sharpens crack edges.

```
L_total = 0.4·CE + 0.4·Dice + 0.2·Boundary
```

### Class-Balanced Sampling

Minority classes (spalling, corrosion, deformation) are oversampled to balance class representation during batching.

## 📖 For More Information

- **Architecture details**: See [../ARCHITECTURE.md](../ARCHITECTURE.md)
- **Model selection rationale**: See [../DECISIONS.md](../DECISIONS.md)
- **API & inference**: See [../README.md](../README.md)
- **What:** Train entire model with layer-wise learning rates
- **Why:** Fine-grained adaptation of encoder features to defect patterns
- **Encoder:** Learning rate 6e-5 (small, preserve ImageNet knowledge)
- **Decoder:** Learning rate 6e-4 (10× larger, fast adaptation)
- **LR Schedule:** Linear warmup (10%) + Cosine decay
- **Output:** `best_model.pth`

## 📈 Data Requirements

Your `../roboflow_data/` folder must contain:

```
roboflow_data/
├── image_0000005.jpg
├── image_0000005.xml  (VOC-style bounding boxes)
├── image_0000021.jpg
├── image_0000021.xml
└── ... more image/xml pairs
```

**VOC-XML Format Example:**
```xml
<annotation>
  <filename>image_0000005.jpg</filename>
  <size>
    <width>1920</width>
    <height>1080</height>
  </size>
  <object>
    <name>crack</name>  <!-- one of: crack, spalling, corrosion, deformation -->
    <bndbox>
      <xmin>100</xmin>
      <ymin>200</ymin>
      <xmax>400</xmax>
      <ymax>500</ymax>
    </bndbox>
  </object>
</annotation>
```

## 🎯 Training Arguments

```
--data_dir           Dataset location (default: ../roboflow_data)
--out_dir            Checkpoint save location (default: ../checkpoints)
--batch_size         Batch size for training (default: 4)
--warmup_epochs      Stage 1 duration in epochs (default: 5)
--main_epochs        Stage 2 duration in epochs (default: 45)
--threshold          Sigmoid threshold for binary masks (default: 0.45)
--num_workers        DataLoader workers (auto-detected: 4 Linux, 2 Windows)
--resume             Path to checkpoint to resume from
--seed               Random seed (default: 42)
```

## 📊 Output Files

Training creates:

```
../checkpoints/
├── stage1_final.pth              # End of decoder warmup
├── best_model.pth                # Best validation mIoU from Stage 2
├── final.pth                      # Last epoch of Stage 2
└── training_log.json              # Metrics per epoch
```

## 🧪 Evaluate Model

```bash
python evaluate.py \
  --data_dir ../roboflow_data \
  --checkpoint ../checkpoints/best_model.pth \
  --save_visuals
```

This will:
- Compute per-class IoU & F1 metrics
- Save qualitative overlays to `eval_output/`
- Print results to console

## ⚡ Performance

| Hardware | Training Time (50 epochs) | Inference (single image) |
|----------|---------------------------|--------------------------|
| NVIDIA A100 | ~30 min | ~8 ms |
| NVIDIA V100 | ~45 min | ~12 ms |
| NVIDIA T4 | ~2 hours | ~32 ms |
| CPU (4-core) | ~8 hours | ~250 ms |

## 🔒 Hyperparameters

Key tuning parameters (editable in train.py):

```python
# Loss weights
bce_w = 0.4          # Cross-entropy weight
dice_w = 0.4         # Dice loss weight
boundary_w = 0.2     # Boundary loss weight

# Stage 1 (decoder warmup)
stage1_lr = 6e-4     # Decoder learning rate
stage1_epochs = 5    # Warm-up duration

# Stage 2 (full fine-tuning)
encoder_lr = 6e-5    # Encoder learning rate (10× smaller)
decoder_lr = 6e-4    # Decoder learning rate (unchanged)
stage2_epochs = 45   # Main training duration
warmup_fraction = 0.10  # Fraction of Stage 2 for LR warmup

# Data
input_size = 512     # Input resolution (512×512)
batch_size = 4       # Batch size per GPU
augmentation = True  # Enable data augmentation
```

## 🆘 Troubleshooting

### "FileNotFoundError: roboflow_data not found"

Make sure your dataset is in the parent directory:
```
claude/
├── roboflow_data/      ← Dataset should be here
├── checkpoints/        ← Or will be created here
└── model_training/
    └── train.py
```

### "CUDA out of memory"

Reduce batch size:
```bash
python train.py --batch_size 2
```

### "No images found in dataset"

Check XML files have valid image references:
```bash
# Ensure images exist for each XML
ls roboflow_data/*.jpg roboflow_data/*.xml
```

### Training is very slow

Check:
1. GPU is being used: `nvidia-smi` should show Python process
2. Increase batch size (if memory allows): `--batch_size 8`
3. Reduce number of DataLoader workers: `--num_workers 0`

## 📚 References

- **Model:** [SegFormer Paper](https://arxiv.org/abs/2105.15203) (Xie et al., 2021)
- **Dataset:** [CODEBRIM](https://github.com/kudzuai/CODEBRIM)
- **Backbone:** [MiT-B2 ImageNet Pretrained](https://huggingface.co/nvidia/mit-b2)

## 📋 Next Steps

1. **Train model:** `python quick_train.py` (takes ~2 hours on GPU)
2. **Evaluate:** `python evaluate.py`
3. **Deploy:** Copy `../checkpoints/best_model.pth` to API

The trained model will automatically be used by the API when placed at:
```
../checkpoints/best_model.pth
```

---

**Happy training! 🚀**
