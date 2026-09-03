# Shortcut-Controlled DS-CNN for HPGe Pulse-Shape Discrimination

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the reproduction code, model architecture, pairing manifests, and pretrained weights for the **Depthwise-Separable Convolutional Neural Network (DS-CNN)** designed for pulse-shape discrimination (PSD) and digital Compton suppression in conventional coaxial HPGe detectors under strict energy-shortcut controls.

---

## Model Overview

* **Architecture:** 1D Depthwise-Separable CNN (DS-CNN) with 4 residual blocks
* **Parameters:** 22,753 trainable weights
* **Input Tensor:** $\mathbf{X} = [Q(t), I(t)]^T \in \mathbb{R}^{2 \times 750}$ ($3.0\,\mu\text{s}$ at 250 MS/s), RMS-normalized
* **Computational Cost:** 2,134,104 multiply-accumulate (MAC) operations per waveform
* **Checkpoint SHA-256:** `cdb90decef73e1ac5ac846df7f6f2f7642ab2d60b912e648602d67696a089677`

---

## Repository Contents

```text
.
├── outputs/
│   ├── experiments/
│   │   ├── strict_ds_cnn_reproducibility_20260826/
│   │   │   └── three_peak_equal_weight/seed_20260825/
│   │   │       └── ds_cnn_best.pt              # Trained DS-CNN model weights
│   │   └── paper1_equal_weight_ds_cnn_20260826/
│   │       ├── held_out/held_out_scores.npz    # Precomputed evaluation scores
│   │       ├── th232_global_threshold/         # Global Pareto threshold results
│   │       └── th232_peak_relative_detection_limit/ # Detection limit results
│   └── labels/
│       └── three_peak_positive_polarity_20260820/
│           ├── label_pairs_train.csv           # 1-to-1 energy-matched training pairs
│           ├── label_pairs_validation.csv      # 17,298 held-out validation pairs
│           └── label_dataset_manifest.json     # Split definitions and metadata
├── scripts/                                    # Training, evaluation, and baseline scripts
├── src/                                        # DS-CNN PyTorch model definitions
├── tests/                                      # Test suite
└── requirements.txt                            # Python dependencies
```

---

## Setup & Dependencies

```bash
git clone git@github.com:thnam-dnri/shortcut_control_PSD.git
cd shortcut_control_PSD

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Model Evaluation & Reproduction

### 1. Evaluate the Trained Model on Held-Out Validation Pairs
Evaluate `ds_cnn_best.pt` against the 17,298 held-out energy-matched pairs ($|\Delta E| < 0.5\,\text{keV}$ across $^{133}\text{Ba}$ 356 keV, $^{22}\text{Na}$ 511 keV, and $^{137}\text{Cs}$ 662 keV):
```bash
python scripts/evaluate_paper1_equal_weight_ds_cnn.py
```
Expected performance:
* **Macro AUROC:** 0.662 (95% CI: [0.656, 0.668])
* **Pooled AUROC:** 0.657 (95% CI: [0.651, 0.663])

### 2. Compare Against the Classical Scalar A/E Baseline
Evaluate conventional current-amplitude-over-energy ($A/E$) on identical held-out matched pairs:
```bash
python scripts/evaluate_traditional_ae.py
```
Expected performance:
* **Macro AUROC:** 0.558 (95% CI: [0.552, 0.564])
* **Pooled AUROC:** 0.559 (95% CI: [0.553, 0.565])
* **Paired DeLong test vs. DS-CNN:** $Z = 24.1, p < 10^{-15}$

### 3. Verify Energy-Shortcut Collapse
Verify how 0.5-keV energy matching collapses the unconstrained energy shortcut from 0.848 AUROC to pure chance (0.501 AUROC):
```bash
python scripts/test_audit_baseline_shortcuts.py
```

### 4. Retrain the Model from Scratch
To reproduce the training run under the exact seed (seed `20260825`):
```bash
python scripts/train_strict_ds_cnn_reproducibility.py
```

---

## License

MIT License.
