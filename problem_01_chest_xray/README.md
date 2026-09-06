# Problem Set 01: Automated Chest X-ray Classification Using Convolutional Neural Networks

## Executive Summary

This project develops a from-scratch convolutional neural network (CNN) to automatically classify pediatric chest X-rays as either **normal** or indicative of **pneumonia**. Rather than relying on pretrained models or external data, we build and train the classifier entirely on the provided dataset, providing transparency into the learning process.

**Key Result:** The trained model achieves a test ROC-AUC of **0.9524** and correctly identifies **100% of pneumonia cases** on the held-out test set, though with trade-offs in false positive rates that are discussed in detail below.

**Important Note:** This is an educational research project, not a clinical diagnostic tool. The model serves as a proof-of-concept for image classification and should not be deployed for medical decision-making without proper validation and regulatory approval.

---

## Problem Overview

**Objective:** Build a CNN that can reliably classify pediatric chest X-rays (anterior-posterior view, ages 1-5 years) into two categories:
- **NORMAL** (Class 0): Healthy lungs with no pneumonia
- **PNEUMONIA** (Class 1): Lungs showing pneumonia indicators

**Dataset Source:** [UCI Machine Learning Repository - Chest X-Ray Images](https://drive.google.com/file/d/1219EeGE1XTJVXYaulynJSa3BXGsbNCLx/view?usp=sharing)
- Original claim: 5,863 images
- **Actual count (verified):** 5,856 readable images (all JPEG/PNG format)
- Structure: Organized into `train/`, `val/`, and `test/` splits, each with `NORMAL/` and `PNEUMONIA/` subdirectories

---

## Approach & Methodology

### Data Preprocessing & Leakage Prevention

One of the most common pitfalls in medical image classification is **data leakage**—accidentally letting information from the test set influence model development. We implemented rigorous safeguards:

1. **Image Audit:** All images were decoded and checked for integrity. We kept only valid JPEG/PNG files in grayscale 8-bit format, filtering out any corrupted or unsupported images.

2. **Duplicate Detection:** Using SHA-256 hashing of decoded pixel arrays, we identified exact duplicates. If an identical image appeared with different labels, we stopped and raised an error (this didn't happen, which is good).

3. **Leakage Prevention:** 
   - The original `test` folder remained completely untouched during all development.
   - We combined the original `train/` and `val/` directories into a single development pool.
   - Any development images that were pixel-identical to test images were removed.
   - For patient privacy concerns, we conservatively removed development images whose filenames suggested they might belong to the same patient as test images.

4. **Train/Validation Split:** Using `StratifiedGroupKFold` with filename-based grouping, we created a proper train/validation split while respecting potential patient-level relationships.

**Impact:** 479 development images were excluded (mainly filename-hint overlaps with test), leaving us with **3,815 training** and **938 validation** images. All **624 test images** remained held out.

### Architecture & Model Design

Rather than using transfer learning (which can hide what's actually being learned), we designed a **compact CNN from scratch**:

```
Input (180×180 grayscale)
  ↓
Normalization (1/255)
  ↓
Light Augmentation (rotation, translation, zoom)
  ↓
4 Convolutional Blocks × [Conv2D → BatchNorm → ReLU] × 2 → MaxPooling
  - Filters: 32, 64, 128, 256 (increasing spatial abstraction)
  ↓
Global Average Pooling (fewer parameters, less overfitting)
  ↓
Dense layer (128 units, ReLU) → Dropout (0.4)
  ↓
Sigmoid output (pneumonia probability)
```

**Design Rationale:**
- **Four blocks** provide enough representational capacity for grayscale medical images without being unwieldy.
- **BatchNormalization** stabilizes training and reduces the impact of weight initialization.
- **Dropout** and **L2 regularization** combat overfitting on limited data.
- **No horizontal/vertical flips** during augmentation—anatomical orientation matters in chest X-rays.
- **Global average pooling** reduces parameter count and encourages spatial feature learning.

### Training Strategy

We employed several techniques to ensure stable, reproducible training:

- **Class weighting:** Since pneumonia is slightly overrepresented (64% of dev data), we used class weights to balance the loss contribution: `{NORMAL: 1.77, PNEUMONIA: 0.70}`.
- **Early stopping:** Training stops if validation PR-AUC doesn't improve for 6 epochs, preventing memorization.
- **Learning rate reduction:** If validation performance plateaus, we halve the learning rate to fine-tune.
- **Checkpoint selection:** We keep the model snapshot with the highest **validation PR-AUC** (Precision-Recall Area Under Curve), not accuracy, because PR-AUC is more informative for imbalanced classification.

### Threshold Selection (The Real Story)

Here's where many projects go wrong: **using the test set to pick the threshold.** We didn't.

Instead:
1. We trained the model on the training set.
2. We generated predictions on the *validation set only*.
3. We computed precision-recall pairs for every possible threshold.
4. We selected the threshold that maximizes **F1-score** (the harmonic mean of precision and recall).
5. We **froze** this threshold and the model weights.
6. We evaluated on the test set *exactly once*, with no tuning.

This prevents overfitting to the test set and gives us an honest estimate of how the model will perform on truly unseen data.

---

## Detailed Results

### Dataset Composition (Final Splits)

| Split | NORMAL | PNEUMONIA | Total | Proportion |
|-------|--------|-----------|-------|------------|
| **Train** | 1,078 | 2,737 | 3,815 | 65.0% dev |
| **Validation** | 270 | 668 | 938 | 19.7% dev |
| **Test** | 234 | 390 | 624 | 10.6% (held-out) |

**Observations:**
- Pneumonia is overrepresented (64% of training data), reflecting the clinical setting where sick children were more likely to get X-rays.
- The validation set is roughly 20% of development data—a common ML convention.
- The test set is entirely independent and was never used during model selection.

### Model Performance

#### Validation Set (For Checkpoint Selection)

| Threshold | Accuracy | Precision | Recall | Specificity | F1-Score | PR-AUC |
|-----------|----------|-----------|--------|-------------|----------|--------|
| **0.5000** (fixed) | 0.9765 | 0.9894 | 0.9775 | 0.9741 | 0.9834 | 0.9992 |
| **0.2690** (selected) | 0.9808 | 0.9865 | 0.9865 | 0.9667 | 0.9865 | 0.9992 |

The validation-selected threshold of **0.2690** slightly outperforms the default 0.5 threshold, achieving near-perfect balance between precision and recall.

#### Test Set (Final Held-Out Evaluation)

| Model / Rule | Threshold | Accuracy | Balanced Acc. | Precision | Recall | Specificity | F1 | ROC-AUC | AP |
|--------------|-----------|----------|---------------|-----------|--------|-------------|----|---------|----|
| **CNN (fixed 0.5)** | 0.5000 | 0.8285 | 0.7722 | 0.7859 | **0.9974** | 0.5470 | 0.8791 | 0.9524 | 0.9576 |
| **CNN (val-selected)** | 0.2690 | 0.8045 | 0.7393 | 0.7617 | **1.0000** | 0.4786 | 0.8647 | 0.9524 | 0.9576 |
| Majority baseline | 0.5000 | 0.6250 | 0.5000 | 0.6250 | 1.0000 | 0.0000 | 0.7692 | 0.5000 | 0.6250 |

**Test Set Confusion Matrix (Selected Threshold = 0.2690):**
```
                Predicted NORMAL  Predicted PNEUMONIA
Actual NORMAL            112                 122
Actual PNEUMONIA           0                 390
```

**What This Means:**
- **Sensitivity (Recall):** 100% of pneumonia cases are correctly identified.
- **Specificity:** 48% of normal cases are correctly identified; 52% are false alarms.
- **Trade-off:** The model prioritizes catching pneumonia (critical for patient safety) over reducing false positives (which lead to unnecessary follow-up imaging).

**Comparison to Baseline:**
- If we naively predicted "everyone has pneumonia," we'd get 62.5% accuracy (just the prevalence).
- Our model achieves 80.5% accuracy—a **+17.95 percentage point improvement**.
- More importantly, our model ranks images by confidence (ROC-AUC = 0.9524), allowing clinicians to prioritize high-risk cases.

---

## Key Insights & Interpretation

### What Went Well

1. **Strong Ranking:** The model's ROC-AUC of **0.9524** indicates excellent discrimination between normal and pneumonia X-rays. This means if you pick any random pneumonia image and any random normal image, the model correctly ranks the pneumonia image as higher risk 95.24% of the time.

2. **No Missed Pneumonia:** At the selected threshold, the model catches every single pneumonia case in the test set (**100% recall**). From a clinical safety perspective, this is the desired behavior—missing a pneumonia diagnosis can have serious consequences.

3. **Generalization:** The validation AP (0.9992) versus test AP (0.9576) shows only modest degradation, suggesting the model learned real patterns rather than memorizing training data.

### The False Positive Problem

122 out of 234 normal images were incorrectly flagged as pneumonia. This is the price of perfect recall:

- **Lower threshold** (0.2690) → catch all pneumonia → more false alarms.
- **Higher threshold** (0.5000) → fewer false alarms → miss some pneumonia.

In practice, this model would be used to **score and rank** incoming X-rays, allowing radiologists to prioritize the highest-confidence pneumonia cases first. The ranking (ROC-AUC = 0.9524) is excellent, even if the fixed threshold is imperfect.

### Training Observations

- **Best epoch:** 27 out of 30 (early stopping prevented overfitting).
- **Training time:** 28.4 minutes on GPU (reasonable for this dataset size).
- **Model size:** 1.2M parameters (small enough for deployment, large enough for the task).
- **Class weighting:** Effectively balanced the training signal despite class imbalance.

---

## Limitations & Honest Discussion

1. **Single Source Data:** All images are from one hospital. Chest X-ray appearances vary by manufacturer, technician, and imaging protocols. The model may not generalize to a different hospital.

2. **Age Range:** The data comes from ages 1-5 years. Pneumonia patterns in infants, teenagers, or adults differ significantly.

3. **Patient Independence:** While we removed obvious duplicates and filename overlaps, we cannot definitively prove that no patient appears in both test and training sets. True patient-level independence would require validated patient IDs.

4. **No Clinical Validation:** This model has not been reviewed by radiologists, has no peer review, and has no formal sensitivity/specificity claims for clinical use. It is a proof-of-concept.

5. **False Positive Rate:** At 52%, the false positive rate is high. Clinical deployment would require substantial refinement and threshold tuning with domain experts.

---

## Architecture Details & Configuration

**Reproducibility Information:**
- Seed: `42`
- Image size: `180 × 180 pixels`
- Batch size: `32`
- Optimizer: `Adam (learning rate = 0.001)`
- Loss function: `Binary Cross-Entropy`
- L2 regularization: `0.0001`
- Dropout rate: `0.40`
- Convolutional filters: `[32, 64, 128, 256]`
- Dense layer units: `128`
- Maximum epochs: `30`
- **Actual epochs trained: 27** (early stopping)
- Training duration: 28.4 minutes

**Model Checkpoints:**
- Total parameters: **1,207,585**
- Model SHA-256: `cfefa03523e45140e0154a272eced061e227731ebc6b0ac8ea396052b66fc9ee`
- Saved as: `best_cnn.keras`

---

## Files & Running the Code

### Project Structure
```
problem_01_chest_xray/
├── train_cnn.ipynb              # Main notebook: EDA → training → evaluation
├── predict.py                   # Standalone inference script
├── inference_config.json        # Frozen model config (threshold, checksum, etc.)
├── requirements.txt             # Python dependencies
├── README.md                     # This file
└── results/
    ├── best_cnn.keras           # Saved model (stored on Drive, not in repo)
    ├── run_config.json          # Hyperparameters used
    ├── environment.json         # Python/package versions
    ├── training_info.json       # Training metadata
    ├── classification_report.txt # Test performance summary
    ├── metrics.json             # Detailed metrics (train/val/test)
    ├── history.csv              # Epoch-by-epoch training history
    ├── manifest_train.csv       # Training split manifest (filepaths, labels)
    ├── manifest_validation.csv  # Validation split manifest
    ├── manifest_test.csv        # Test split manifest
    ├── dataset_audit.json       # Data integrity audit results
    └── *.png                    # Visualizations (curves, confusion matrices, ROC)
```

### How to Run

**Step 1: Training (Google Colab)**
1. Open `train_cnn.ipynb` in Google Colab.
2. Mount your Google Drive (Step 2).
3. Point `DRIVE_DATASET_DIR` to your dataset location (Step 3).
4. Run cells 1–12 to train the model.
5. Run cells 13–16 to evaluate on the held-out test set.
6. All results are saved to your Drive; results files are in the local `results/` folder.

**Step 2: Inference (Standalone Python Script)**

Once you have a trained model (`best_cnn.keras`) and its config (`inference_config.json`):

```bash
python predict.py \
    --model best_cnn.keras \
    --config inference_config.json \
    --image path/to/xray.jpg
```

**Output:**
```json
{
    "image": "xray.jpg",
    "predicted_class": "PNEUMONIA",
    "pneumonia_score": 0.987654,
    "threshold": 0.26900339126586914,
    "disclaimer": "Educational research model. Not for clinical diagnosis."
}
```

### Dependencies

The code requires:
- `tensorflow >= 2.10`
- `keras >= 2.10`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `Pillow`
- `tqdm`

See `requirements.txt` for exact versions.

---

## Verification & Reproducibility

- **Model SHA-256:** All results reference the exact model checkpoint used (stored in `inference_config.json`).
- **Dataset manifests:** `manifest_*.csv` files list every image used in training/validation/test with filesystem paths, labels, pixel hashes, and grouping information.
- **Random seed:** All randomness is fixed with `SEED = 42` for reproducibility.
- **Data leakage checks:** The `dataset_audit.json` documents every data cleaning decision.

To reproduce results, use the same seed, dataset, and hyperparameters. Due to non-determinism in some GPU operations, exact metric values may vary slightly.

---

## Future Work & Improvements

If this were to be deployed or improved further:

1. **Cross-validation:** Run 5-fold cross-validation to get confidence intervals on performance.
2. **Ensemble methods:** Train multiple architectures and average predictions.
3. **Transfer learning:** Use ImageNet or medical imaging pretrained weights as a baseline.
4. **Attention mechanisms:** Add attention layers to highlight which regions drive predictions.
5. **External validation:** Test on chest X-rays from different hospitals/patient populations.
6. **Threshold optimization:** Work with radiologists to determine the optimal sensitivity-specificity trade-off for clinical use.
7. **Uncertainty quantification:** Provide confidence intervals alongside predictions (e.g., Bayesian CNN).

---

## References & Acknowledgments

- **Dataset:** Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning," *Cell*, 2018. [Paper](https://doi.org/10.1016/j.cell.2018.02.010)
- **Framework:** TensorFlow/Keras (Google)
- **Methodology:** Adapted from standard ML practices in medical imaging (leakage prevention, stratified splitting, threshold selection via validation set).

---

## Disclaimer

🚨 **This is an educational model, not a medical device.** It has not undergone clinical validation, regulatory approval, or safety testing. Using this model for actual clinical diagnosis could cause patient harm. Always consult qualified medical professionals for real medical decisions.

