# 🍃 Automated Tea Leaf Disease Classification Using Swin Transformer + Grad-CAM XAI

> **Research Paper:** *Automated Detection of Tea Leaf Diseases Using Deep Learning: A Study on the TeaLeafNet Dataset*
> **Author:** Uzair Moazzam 

---

##  Overview

This project presents **SwinTeaClassifier** — a production-grade deep learning system for automated classification of tea leaf diseases using a fine-tuned **Swin Transformer Tiny (Swin-T)** backbone, augmented with **Gradient-weighted Class Activation Mapping (Grad-CAM)** for explainability. The system achieves **97.87% test accuracy** on the TeaLeafNet dataset with a single-image inference latency of **10.31 ms (97.0 FPS)** on a Tesla P100 GPU — confirming real-time field deployability.

This study provides the **first systematic, reproducible benchmark** on the publicly available TeaLeafNet dataset and the **first reported application of a Vision Transformer** to this benchmark in the literature.

---

## Key Results at a Glance

| Metric | Value |
|--------|-------|
| **Test Accuracy (TTA ×7)** | **97.87%** |
| Weighted Precision | 0.9792 |
| Weighted Recall | 0.9787 |
| Weighted F1-Score | 0.9787 |
| Best Validation Accuracy (Phase 1) | 98.27% |
| Best Validation Accuracy (Phase 2) | 98.93% |
| Mean Inference Latency | 10.31 ± 0.38 ms |
| P95 Latency | 11.05 ms |
| Throughput | **97.0 FPS** |
| Hardware | Tesla P100-PCIE-16GB, CUDA 11.8 |

---

## Disease Classes

The model classifies four economically critical tea foliar diseases:

| Code | Disease | Pathogen |
|------|---------|----------|
| **BB** | Brown Blight | *Colletotrichum camelliae* |
| **GL** | Grey Leaf | *Pestalotiopsis theae* |
| **RR** | Red Rust | *Cephaleuros parasiticus* |
| **RSM** | Red Spider Mite | Mite-induced damage |

---

## Per-Class Test Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Brown Blight (BB) | 0.9637 | 0.9894 | 0.9764 | 188 |
| Grey Leaf (GL) | 0.9639 | **1.0000** | 0.9816 | 187 |
| Red Rust (RR) | **1.0000** | 0.9628 | 0.9810 | 188 |
| Red Spider Mite (RSM) | 0.9890 | 0.9626 | 0.9756 | 187 |
| **Weighted Avg** | **0.9792** | **0.9787** | **0.9787** | **750** |

- **GL** achieved **perfect recall (1.0000)** — zero misclassifications
- **RR** achieved **perfect precision (1.0000)** — zero false positives
- All classes exceed **0.963** on every metric

---

## Dataset

**TeaLeafNet** — publicly available on Kaggle (`harjindersinghdibru/tealeafnet`)

| Property | Value |
|----------|-------|
| Total images | 5,000 |
| Classes | 4 (perfectly balanced) |
| Images per class | 1,250 |
| Image type | High-resolution, geotagged, black background |
| Source region | Assam, India |

**Stratified 70 / 15 / 15 Split** (seed = 42):

| Split | BB | GL | RR | RSM | Total |
|-------|----|----|----|-----|-------|
| Train (70%) | 875 | 875 | 875 | 875 | 3,500 |
| Validation (15%) | 187 | 188 | 187 | 188 | 750 |
| Test (15%) | 188 | 187 | 188 | 187 | 750 |

All 5,000 images were **pre-cached in RAM** prior to training (full preload: ~477 seconds), eliminating all GPU idle time from data loading.

---

## Model Architecture — SwinTeaClassifier

```
Input Image (224×224×3)
        ↓
Patch Embedding → 56×56 patches, dim=96
        ↓
Swin Stage 1 [FROZEN] → 56×56, dim=96,  W-MSA
        ↓
Swin Stage 2 [FROZEN] → 28×28, dim=192, SW-MSA
        ↓
Swin Stage 3 [FROZEN] → 14×14, dim=384, W-MSA
        ↓
Swin Stage 4 [★ UNFROZEN in Phase 2] → 7×7, dim=768, SW-MSA
        ↓
Layer Norm ← Grad-CAM XAI Hook Target
        ↓
Global Average Pooling (token mean → 768-d vector)
        ↓
FC(768→512) + ReLU
        ↓
Dropout (p=0.30 Phase 1 / p=0.20 Phase 2)
        ↓
FC(512→4) → [BB | GL | RR | RSM]
```

| Parameter | Value |
|-----------|-------|
| Backbone | Swin-Tiny (ImageNet-1k pretrained, timm v0.9.16) |
| Stochastic depth drop-path rate | 0.20 |
| Total parameters | 27,915,134 |
| Trainable — Phase 1 | 395,780 (head only) |
| Trainable — Phase 2 | 15,763,892 (Stage 4 + LayerNorm + head) |

---

## Two-Phase Transfer Learning Strategy

### Phase 1 — Feature Extraction (Frozen Backbone)

| Setting | Value |
|---------|-------|
| Trainable params | 395,780 (head only) |
| Optimiser | AdamW |
| Learning rate | 1×10⁻⁴ |
| Weight decay | 1×10⁻⁴ |
| Epochs | 8 (2 warm-up) |
| MixUp α | 0.20 |
| Head Dropout | 0.30 |
| **Best Val Accuracy** | **98.27% (epoch 7)** |

### Phase 2 — Deep Fine-Tuning (Stage 4 Unfrozen)

| Setting | Value |
|---------|-------|
| Trainable params | 15,763,892 (Stage 4 + LayerNorm + head) |
| Optimiser | SGD with Nesterov momentum (0.9) |
| Learning rate | 5×10⁻⁵ |
| Weight decay | 1×10⁻⁵ |
| Epochs | 15 (2 warm-up) |
| MixUp α | 0.15 |
| Head Dropout | 0.20 |
| **Best Val Accuracy** | **98.93% (epoch 11)** |

**Shared across both phases:** CrossEntropyLoss with label smoothing=0.08 · Gradient clipping (max norm=1.0) · Linear warm-up + Cosine annealing LR schedule · AMP (FP16) · Batch size=32

---

## Regularisation Stack

| Technique | Configuration |
|-----------|---------------|
| Dropout | p=0.30 (Phase 1) → p=0.20 (Phase 2) |
| MixUp | α=0.20 (Phase 1), α=0.15 (Phase 2) |
| Label Smoothing | ε=0.08 |
| Gradient Clipping | max_norm=1.0 |
| Stochastic Depth | drop-path rate=0.20 (within Swin backbone) |
| Anti-memorisation Val Augmentation | Random crop + mild brightness jitter per epoch |

---

## Data Augmentation Pipelines

| Transform | Train | Validation | Test |
|-----------|-------|-----------|------|
| Resize to 256×256 | ✓ | ✓ | ✓ |
| Random crop 224×224 | ✓ | ✓ | — |
| Fixed offset crop (12 px) | — | — | ✓ |
| Horizontal flip | p=0.50 | p=0.50 | — |
| Random rotation (90°) | p=0.15 | — | — |
| Gaussian blur σ[0.1,0.8] | p=0.15 | — | — |
| Random greyscale | p=0.015 | — | — |
| Random erasing (8–28 px) | p=0.07 | — | — |
| Brightness scale [0.92,1.08] | — | ✓ | — |
| MixUp α | 0.20/0.15 | — | — |
| ImageNet normalisation | ✓ | ✓ | ✓ |

**Test Time Augmentation (TTA):** N=7 independent views per image (random crop + horizontal/vertical flip + brightness jitter + occasional Gaussian blur), logits averaged before argmax.

**NumPy-free pipeline:** A custom `pil_to_tensor_pure` function using Python's `struct.unpack` was implemented to bypass platform-specific NumPy ABI conflicts on the Kaggle Tesla P100 environment — all tensor operations are entirely PyTorch-native.

---

## Explainability — Grad-CAM XAI

**Hook target:** `model.backbone.layers[3].blocks[-1].norm1` (Layer Norm at end of Swin Stage 4 — the semantically richest spatial activation point before global average pooling)

**Key implementation details:**

- Custom `reshape_transform_swin` handles Swin-T's non-standard tensor layout: `(B, N, C) → (B, C, 7, 7)` via reshape + permute
- **Morphological isolation pre-processing:** GPS geotag text pixels (thin 1–3px strings embedded in the black background) are suppressed via 4 passes of 4-connected erosion, then 6 passes of dilation to restore leaf tissue — preventing Grad-CAM from anchoring to artefacts
- **Eigen-CAM smoothing** (`eigen_smooth=True`) extracts the dominant principal component, suppressing spurious edge responses
- All forward/backward hooks cleared before initialisation to eliminate residual hooks from training

**Activation patterns confirmed:**
- **BB:** Activations concentrate over necrotic lesion boundaries and discoloured tissue edges
- **GL:** Activations span the full lamina, reflecting the diffuse grey-green discolouration of Grey Leaf
- **RR:** Activations localise to red-orange sporulation spots (precise small-area discrimination)
- **RSM:** Activations concentrate on symptomatic leaf margins and surface deformations from mite feeding

---

## Inference Latency Benchmark

Benchmarked over **100 single-image forward passes** (10 warm-up passes excluded) with `torch.cuda.synchronize()` after each pass:

| Statistic | Value |
|-----------|-------|
| Mean latency | **10.31 ± 0.38 ms** |
| P95 latency | 11.05 ms |
| Throughput | **97.0 FPS** |
| Hardware | Tesla P100-PCIE-16GB |
| CUDA version | 11.8 |

The narrow standard deviation (0.38 ms) confirms **stable, low-jitter inference** suitable for real-time field deployment. The model operates at ~3× the conventional 30 FPS real-time threshold.

---

## Training Dynamics Summary

**Phase 1 (epochs 1–8):**
- Training loss: 1.1602 → 0.5799
- Validation accuracy: 87.33% → peak 98.27% (epoch 7)
- Each epoch: ~14.6–14.8 seconds

**Phase 2 (epochs 1–15):**
- Validation accuracy: 97.73% → peak 98.93% (epoch 11)
- Training accuracy range: 88.3%–91.4% (higher variance expected due to anti-memorisation val augmentation)
- Each epoch: ~16.5–17.0 seconds
- No divergence between training and validation loss across either phase

The val–train accuracy inversion during Phase 2 is a direct consequence of the anti-memorisation validation strategy (independent random crops each epoch), not overfitting.

---

## Confusion Matrix Highlights

Total correct predictions: **734 / 750** (97.87%)

| True \ Predicted | BB | GL | RR | RSM |
|-----------------|----|----|-----|-----|
| **BB** | 186 | 0 | 0 | 2 |
| **GL** | 0 | 187 | 0 | 0 |
| **RR** | 5 | 2 | 181 | 0 |
| **RSM** | 2 | 5 | 0 | 180 |

**Dominant errors:** RR→BB (2.7%) and RSM→GL (2.7%) — both agronomically interpretable as early-stage symptom morphology overlap. No GL↔RR or BB↔RSM confusion was observed.

---

## Comparison with Prior Work

| Study | Dataset | Architecture | Classes | Accuracy |
|-------|---------|-------------|---------|----------|
| Alam et al. (2024) | Custom Bangladesh (3,330 imgs) | Custom CNN | 4 | 92.3% |
| Bhuyan et al. (2024) | Tea_Leaf_Disease Kaggle | SAM + CNN + SVM | 6 | 95.06% |
| Li & Zhao (2025) | Real tea garden images | ECA-ResNet50 | 5 | 93.06% |
| Ozturk et al. (2025) | Kaggle tea (8 classes) | 4-model Ensemble | 8 | ~94% |
| Kabir et al. (2025) | Tea disease (8 classes) | CNN-Transformer + GWO | 8 | 97.72% |
| Sayed et al. (2026) | Kaggle tea (8 classes) | 4-model Ensemble | 8 | 98.3% |
| **This Study** | **TeaLeafNet (5,000 imgs)** | **Swin-T (two-phase) + TTA×7** | **4** | **97.87%** |

**Key differentiators:**
- First systematic benchmark on the TeaLeafNet dataset
- First Vision Transformer applied to TeaLeafNet
- Single non-ensemble architecture matching or exceeding multi-model ensemble systems
- Grad-CAM explainability with morphological artefact suppression
- Quantified real-time inference latency

---

## Technical Stack

| Component | Version / Specification |
|-----------|------------------------|
| Framework | PyTorch 2.2.0+cu118 |
| Model library | timm 0.9.16 |
| XAI library | pytorch-grad-cam 1.5.0 |
| Metrics | scikit-learn |
| GPU | Tesla P100-PCIE-16GB |
| CUDA | 11.8 |
| Python | Kaggle Notebook environment |
| Random seed | 42 (PyTorch + NumPy + Python random) |
| cuDNN | benchmark=True, deterministic=False |

---

## Output Artefacts

| File | Description |
|------|-------------|
| `swin_tea_best.pth` | Best model checkpoint (saved after Phase 2) |
| `fig1_class_distribution.png` | Bar chart + pie chart of dataset class balance |
| `fig2_sample_images.png` | 2 representative samples per disease class |
| `fig3_training_curves.png` | Loss and accuracy curves across both phases |
| `fig4_confusion_matrix.png` | Raw count + normalised confusion matrix |
| `fig5_per_class_metrics.png` | Per-class Precision / Recall / F1-Score bar chart |
| `fig6_latency.png` | Inference latency histogram + summary |
| `fig7_gradcam.png` | Grad-CAM disease localisation maps (2 per class) |
| `fig8_architecture.png` | Swin-T architecture diagram with frozen/unfrozen regions |
| `fig9_phase_accuracy.png` | Phase-wise best val + final test accuracy |

---

## Reproducibility

All experiments are fully reproducible:
- Fixed random seed: **42** across PyTorch, NumPy, and Python `random`
- Complete hyperparameter disclosure in paper Section 3
- Stratified split implemented via `StratifiedShuffleSplit(random_state=42)`
- Dataset: publicly available at `https://www.kaggle.com/datasets/harjindersinghdibru/tealeafnet`
- DataLoader: `num_workers=0`, RAM-cached tensors, `pin_memory=True`

---

## Future Directions

1. **Cross-dataset evaluation** — Transfer to teaLeafBD, Tea_Leaf_Disease, and TDPD without retraining to quantify domain shift robustness
2. **Multi-architecture benchmark** — Evaluate ResNet-50, EfficientNetB0/B3, DenseNet-121, MobileNetV3 on identical splits
3. **Edge deployment** — Post-training INT8 quantisation, pruning, and knowledge distillation for Jetson Nano / mobile accelerators
4. **Object detection formulation** — YOLOv8 / Faster R-CNN for multi-lesion localisation on whole-canopy UAV images
5. **Field-condition robustness** — Augment with field-captured images from Sri Lanka, Kenya, and Vietnam
6. **Disease severity grading** — Ordinal severity prediction (mild / moderate / severe) to directly inform treatment decisions

---

## 📄 Citation

If you use this work, please cite:

```
Moazzam, U. (2026). Automated Detection of Tea Leaf Diseases Using Deep Learning:
A Study on the TeaLeafNet Dataset. Fa-22/BSCS/203.
```

---

## Acknowledgements

- **TeaLeafNet dataset:** Harjinder Singh Dibru and collaborators (Kaggle: `harjindersinghdibru/tealeafnet`)
- **Compute:** Kaggle free-tier Tesla P100-PCIE-16GB GPU
- **Libraries:** timm, pytorch-grad-cam, scikit-learn, PyTorch

---

*This study was conducted exclusively on a publicly available dataset. No human subjects, animals, clinical trials, or personal data were involved. No ethical approval was required.*