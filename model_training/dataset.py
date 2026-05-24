"""
dataset.py — CODEBRIM multi-label segmentation dataset

Key fixes and upgrades over gem/dataset.py:
  1. Stratified multi-label split (skmultilearn) instead of random split
     → guarantees rare classes (deformation) appear in both train & val
  2. Per-class Otsu with class-aware inversion logic
     → corrosion uses THRESH_BINARY (not INV) because rust is brighter
  3. Stronger augmentation pipeline: ElasticTransform + GridDistortion added
     → critical for crack shape variance on small dataset (~1,590 images)
  4. Graceful fallback when skmultilearn is not installed
  5. Dataset statistics printed on load (class frequencies)
  6. Returns float32 masks clamped to [0,1] — guards against Otsu edge cases
"""

import os
import cv2
import glob
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


# ── Class schema ───────────────────────────────────────────────────────────────
# Index:  0=efflorescence  1=corrosion  2=crack  3=spalling  4=exposed_bars
CLASS_NAMES = ["efflorescence", "corrosion", "crack", "spalling", "exposed_bars"]

CLASS_MAP = {
    # efflorescence (white mineral deposits — BRIGHTER than background)
    "0 efflorescence": 0, "efflorescence": 0,
    # corrosion / rust stain (orange/brown — BRIGHTER than grey concrete)
    "1 corrosionstain": 1, "corrosionstain": 1, "corrosion stain": 1,
    "corrosion": 1, "rust": 1,
    # crack (dark lines — DARKER than background)
    "2 crack": 2, "crack": 2, "cracks": 2,
    # spalling (dark exposed aggregate — DARKER than background)
    "3 spallation": 3, "spallation": 3, "spalling": 3,
    # exposed rebar (dark metal — DARKER than background)
    "4 exposedbars": 4, "exposedbars": 4, "exposed_bars": 4,
    "exposed bars": 4, "exposed rebar": 4, "rebar": 4,
}

# Classes where defect is DARKER than background → use THRESH_BINARY_INV
# Classes where defect is BRIGHTER than background → use THRESH_BINARY
INVERT_OTSU = {0: False, 1: False, 2: True, 3: True, 4: True}


# ── Mask generation helpers ────────────────────────────────────────────────────

def _otsu_refine(box_region: np.ndarray, class_idx: int) -> np.ndarray:
    """
    Apply class-aware Otsu thresholding inside a bounding-box crop.

    Returns a float32 mask in [0,1] the same spatial size as box_region.

    Notes
    -----
    - Corrosion (rust) is orange/brown = brighter than grey concrete,
      so we do NOT invert the Otsu result for that class.
    - For all other classes (cracks, spalling, deformation) the defect
      is darker than the surrounding concrete, so we invert.
    - We blur slightly before thresholding to suppress JPEG compression
      noise that would otherwise create isolated single-pixel blobs.
    """
    gray = cv2.cvtColor(box_region, cv2.COLOR_RGB2GRAY)

    # Mild Gaussian blur to suppress noise before Otsu
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    flag = cv2.THRESH_BINARY_INV if INVERT_OTSU[class_idx] else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)

    # Remove tiny noise blobs (< 20px²) with morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return (binary / 255.0).astype(np.float32)


# ── Dataset ────────────────────────────────────────────────────────────────────

class CodebrimDataset(Dataset):
    """
    CODEBRIM dataset — VOC-style XML annotations, multi-label pixel masks.

    Parameters
    ----------
    data_dir : str
        Directory containing *.jpg / *.png images and matching *.xml files.
    file_basenames : list[str]
        Subset of file base names (no extension) to include in this split.
    is_train : bool
        If True, applies full augmentation pipeline.
    """

    def __init__(self, data_dir: str, file_basenames: list, is_train: bool = True):
        self.data_dir      = data_dir
        self.file_basenames = file_basenames
        self.is_train      = is_train
        self.class_map     = CLASS_MAP

        # ── Augmentation pipelines ─────────────────────────────────────────────
        if is_train:
            self.transform = A.Compose([
                A.Resize(512, 512, interpolation=cv2.INTER_LINEAR),

                # Geometric augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1,
                    rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.4
                ),

                # Elastic / grid deformations — critical for crack shape variance
                A.ElasticTransform(
                    alpha=60, sigma=8, p=0.25
                ),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),

                # Photometric augmentations
                A.ColorJitter(
                    brightness=0.25, contrast=0.25,
                    saturation=0.2, hue=0.05, p=0.5
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),

                # Occasional blur / sharpening to simulate camera focus variance
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.Sharpen(alpha=(0.1, 0.3), p=1.0),
                ], p=0.2),

                # Normalise with ImageNet stats (SegFormer pretrained on ImageNet)
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(512, 512, interpolation=cv2.INTER_LINEAR),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.file_basenames)

    def __getitem__(self, idx: int):
        base = self.file_basenames[idx]

        # ── Resolve image path (jpg preferred, png fallback) ───────────────────
        img_path = os.path.join(self.data_dir, f"{base}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.data_dir, f"{base}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found for base '{base}' in {self.data_dir}")

        xml_path = os.path.join(self.data_dir, f"{base}.xml")

        # ── Load image ─────────────────────────────────────────────────────────
        image = cv2.imread(img_path)
        if image is None:
            raise IOError(f"cv2.imread failed for {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w  = image.shape[:2]

        # ── Build multi-label mask [H, W, 4] ──────────────────────────────────
        # dtype float32 in [0, 1]; channel c = probability pixel belongs to class c
        multi_mask = np.zeros((h, w, 5), dtype=np.float32)

        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall("object"):
                name_elem = obj.find("name")
                if name_elem is None:
                    continue
                label = name_elem.text.strip().lower()
                if label not in self.class_map:
                    continue

                ch = self.class_map[label]

                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue

                xmin = int(float(bndbox.find("xmin").text))
                ymin = int(float(bndbox.find("ymin").text))
                xmax = int(float(bndbox.find("xmax").text))
                ymax = int(float(bndbox.find("ymax").text))

                # Clamp to image boundaries
                xmin = max(0, min(xmin, w - 1))
                xmax = max(0, min(xmax, w))
                ymin = max(0, min(ymin, h - 1))
                ymax = max(0, min(ymax, h))

                if xmax <= xmin or ymax <= ymin:
                    continue

                # ── Class-aware Otsu refinement inside bbox ────────────────────
                box_region    = image[ymin:ymax, xmin:xmax]
                refined_mask  = _otsu_refine(box_region, ch)

                # max-pool handles overlapping annotations of the same class
                multi_mask[ymin:ymax, xmin:xmax, ch] = np.maximum(
                    multi_mask[ymin:ymax, xmin:xmax, ch],
                    refined_mask
                )

        # Clamp to [0, 1] as a safety guard
        multi_mask = np.clip(multi_mask, 0.0, 1.0)

        # ── Apply transforms ───────────────────────────────────────────────────
        augmented = self.transform(image=image, mask=multi_mask)
        image_t   = augmented["image"]                    # [3, 512, 512]
        mask_t    = augmented["mask"].permute(2, 0, 1)    # [4, 512, 512]

        return image_t, mask_t


# ── Dataset statistics ─────────────────────────────────────────────────────────

def compute_class_frequencies(data_dir: str, basenames: list) -> np.ndarray:
    """
    Compute per-class presence frequency across all samples.
    Used to inform pos_weight selection and print dataset stats.
    Returns array of shape [5] with fraction of images containing each class.
    """
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int32)
    for base in basenames:
        xml_path = os.path.join(data_dir, f"{base}.xml")
        if not os.path.exists(xml_path):
            continue
        tree = ET.parse(xml_path)
        root = tree.getroot()
        found = set()
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is None:
                continue
            label = name_elem.text.strip().lower()
            if label in CLASS_MAP:
                found.add(CLASS_MAP[label])
        for ch in found:
            counts[ch] += 1

    total = max(len(basenames), 1)
    return counts / total


# ── Multi-label stratified split ──────────────────────────────────────────────

def _build_label_matrix(data_dir: str, basenames: list) -> np.ndarray:
    """
    Build binary label matrix [N, 4] for stratified splitting.
    Entry (i, c) = 1 if sample i contains class c.
    """
    Y = np.zeros((len(basenames), len(CLASS_NAMES)), dtype=np.int32)
    for i, base in enumerate(basenames):
        xml_path = os.path.join(data_dir, f"{base}.xml")
        if not os.path.exists(xml_path):
            continue
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is None:
                continue
            label = name_elem.text.strip().lower()
            if label in CLASS_MAP:
                Y[i, CLASS_MAP[label]] = 1
    return Y


def stratified_multilabel_split(
    data_dir: str,
    basenames: list,
    val_split: float = 0.15,
    test_split: float = 0.10,
    seed: int = 42,
):
    """
    Split basenames into train / val / test with multi-label stratification.

    Tries skmultilearn.model_selection.iterative_train_test_split first.
    Falls back to sklearn random split if skmultilearn is not installed.

    Returns
    -------
    train_bases, val_bases, test_bases : list[str]
    """
    n = len(basenames)
    basenames_arr = np.array(basenames)
    Y = _build_label_matrix(data_dir, basenames)

    try:
        from skmultilearn.model_selection import iterative_train_test_split

        # Split off test set first
        idx_all   = np.arange(n).reshape(-1, 1)
        idx_dev, _, idx_test, _ = iterative_train_test_split(
            idx_all, Y, test_size=test_split
        )
        idx_dev  = idx_dev.ravel()
        idx_test = idx_test.ravel()

        # Split remaining into train / val
        val_ratio_of_dev = val_split / (1.0 - test_split)
        idx_train, _, idx_val, _ = iterative_train_test_split(
            idx_dev.reshape(-1, 1), Y[idx_dev], test_size=val_ratio_of_dev
        )
        idx_train = idx_train.ravel()
        idx_val   = idx_val.ravel()

        print("[Split] Using iterative multi-label stratification (skmultilearn).")

    except ImportError:
        print(
            "[Split] skmultilearn not found — falling back to random split.\n"
            "        Install with: pip install scikit-multilearn\n"
            "        This may leave rare classes under-represented in val/test."
        )
        rng = np.random.default_rng(seed)
        idx_all   = rng.permutation(n)
        n_test    = max(1, int(n * test_split))
        n_val     = max(1, int(n * val_split))
        idx_test  = idx_all[:n_test]
        idx_val   = idx_all[n_test:n_test + n_val]
        idx_train = idx_all[n_test + n_val:]

    train_bases = basenames_arr[idx_train].tolist()
    val_bases   = basenames_arr[idx_val].tolist()
    test_bases  = basenames_arr[idx_test].tolist()

    return train_bases, val_bases, test_bases


# ── DataLoader factory ─────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir:   str,
    batch_size: int   = 4,
    val_split:  float = 0.15,
    test_split: float = 0.10,
    seed:       int   = 42,
    num_workers: int  = None,
):
    """
    Scan data_dir for XML annotations, split, build datasets and DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, class_weights : torch.Tensor [5]
        class_weights are reciprocal-frequency weights suitable for pos_weight.
    """
    # Discover all annotated samples
    all_xmls   = glob.glob(os.path.join(data_dir, "*.xml"))
    basenames  = sorted([os.path.splitext(os.path.basename(x))[0] for x in all_xmls])

    if len(basenames) == 0:
        raise RuntimeError(f"No XML annotation files found in: {data_dir}")

    print(f"\n[Dataset] Found {len(basenames)} annotated samples in {data_dir}")

    # Stratified split
    train_bases, val_bases, test_bases = stratified_multilabel_split(
        data_dir, basenames, val_split=val_split, test_split=test_split, seed=seed
    )
    print(f"[Dataset] Split → train={len(train_bases)}  val={len(val_bases)}  test={len(test_bases)}")

    # Print per-class frequencies to verify balance
    train_freq = compute_class_frequencies(data_dir, train_bases)
    val_freq   = compute_class_frequencies(data_dir, val_bases)
    print("\n[Dataset] Class presence frequency (fraction of images):")
    print(f"  {'Class':<14} {'Train':>8} {'Val':>8}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<14} {train_freq[i]:>7.1%} {val_freq[i]:>7.1%}")

    # Compute pos_weight = (1 - freq) / freq  → higher weight for rare classes
    # Clipped to [1.0, 10.0] to prevent extreme values on very rare classes
    eps    = 1e-6
    freq   = np.clip(train_freq, eps, 1 - eps)
    pw_arr = np.clip((1.0 - freq) / freq, 1.0, 10.0).astype(np.float32)
    class_weights = torch.tensor(pw_arr)
    print(f"\n[Dataset] Computed pos_weight: {pw_arr.tolist()}")

    # Build datasets
    train_ds = CodebrimDataset(data_dir, train_bases, is_train=True)
    val_ds   = CodebrimDataset(data_dir, val_bases,   is_train=False)
    test_ds  = CodebrimDataset(data_dir, test_bases,  is_train=False)

    # Auto workers: 4 on Linux/Mac, 2 on Windows
    nw = num_workers if num_workers is not None else (4 if os.name != "nt" else 2)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/codebrim"
    print(f"Smoke-testing dataset pipeline on: {data_dir}")

    train_loader, val_loader, test_loader, cw = get_dataloaders(
        data_dir, batch_size=2
    )
    images, masks = next(iter(train_loader))
    print(f"\nBatch shapes — images: {images.shape}  masks: {masks.shape}")
    print(f"Image dtype/range: {images.dtype}  [{images.min():.2f}, {images.max():.2f}]")
    print(f"Mask  dtype/range: {masks.dtype}   [{masks.min():.2f}, {masks.max():.2f}]")
    print(f"Class weights: {cw.tolist()}")
    print("\nDataset pipeline OK.")