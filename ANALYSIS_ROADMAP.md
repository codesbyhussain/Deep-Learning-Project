# MultiROCKET ECG Classification — Analysis Roadmap (2 × 2 Ablation)

**Objective:** Evaluate dimensionality reduction strategy (Autoencoder vs Structured Pooling) and classifier architecture (MLP vs FT-Transformer) on a fixed MultiROCKET feature representation for 4-class ECG classification.

---

## 1. Data Verification & Cohort Definition

**Targets**

- [ ] Confirm dataset integrity (≈45k recordings, 12 leads, 10 s each)
- [ ] Verify sampling rate (e.g., 500 Hz → 5,000 samples/lead)
- [ ] Confirm one recording per patient (no duplicates)
- [ ] Validate label mapping for the 4 target diagnoses
- [ ] Ensure labels align with records (no missing targets)
- [ ] Determine task type (multiclass vs multilabel)
- [ ] Compute class counts and imbalance ratios
- [ ] Identify extreme minority classes

**Outputs**

- [ ] Table of class frequencies
- [ ] Basic dataset statistics
- [ ] Confirmation of usable sample count

---

## 2. Train / Validation / Test Partitioning

**Targets**

- [ ] Create stratified 60 / 20 / 20 split
- [ ] Use fixed random seed
- [ ] Ensure label distribution consistency across splits
- [ ] Prevent leakage (no overlap)
- [ ] Serialize split indices once

**Outputs**

- [ ] Saved split indices
- [ ] Distribution comparison table across splits
- [ ] Reproducible partition for all experiments

---

## 3. MultiROCKET Feature Extraction (Shared Upstream Representation)

**Targets**

- [ ] Configure MultiROCKET with fixed seed
- [ ] Choose kernel count (e.g., K = 2048 → ~16,384 features)
- [ ] Fit on training data only
- [ ] Transform train/val/test identically
- [ ] Process in mini-batches to avoid RAM overflow
- [ ] Store features as float32 disk-backed arrays
- [ ] Persist fitted transformer

**Validation Checks**

- [ ] Confirm feature dimensionality matches expectation
- [ ] Confirm no NaNs or infinities
- [ ] Verify reproducibility across runs

**Outputs**

- [ ] Memmapped feature matrices for each split
- [ ] Saved transformer object

---

## 4. Feature Scaling

**Targets**

- [ ] Fit scaler on training features only
- [ ] Apply identical transform to val/test
- [ ] Verify numerical stability
- [ ] Persist scaler

**Outputs**

- [ ] Scaled feature datasets
- [ ] Saved scaler

---

## 5. Dimensionality Reduction — Path A (Autoencoder)

**Targets**

- [ ] Design shallow feedforward autoencoder
- [ ] Select latent dimension (d ≪ 16k; e.g., 128–512)
- [ ] Train using reconstruction loss only
- [ ] Monitor convergence (avoid collapse)
- [ ] Prevent overfitting (validation reconstruction error)
- [ ] Retain encoder only

**Validation Checks**

- [ ] Reconstruction error reasonable
- [ ] Latent vectors finite and stable
- [ ] Latent dimensionality as specified

**Outputs**

- [ ] Latent representations for train/val/test
- [ ] Saved encoder model

---

## 6. Dimensionality Reduction — Path B (Structured Pooling)

**Targets**

- [ ] Define grouping by kernel × statistic × origin
- [ ] Implement deterministic pooling operator
- [ ] Apply identical procedure across splits
- [ ] Verify resulting dimensionality

**Validation Checks**

- [ ] Output shape consistent with design
- [ ] No information leakage
- [ ] Numerical stability

**Outputs**

- [ ] Pooled feature datasets

---

## 7. Classifier Training — MLP (A1 & B1)

**Targets**

- [ ] Configure shallow fully connected network
- [ ] Train on AE latent features (A1)
- [ ] Train on pooled features (B1)
- [ ] Use same training protocol for both
- [ ] Tune learning rate / epochs if necessary
- [ ] Prevent overfitting (validation monitoring)

**Outputs**

- [ ] Trained MLP models for A1 and B1
- [ ] Training histories

---

## 8. Classifier Training — FT-Transformer (A2 & B2)

**Targets**

- [ ] Configure feature-token transformer
- [ ] Verify compatibility with dense inputs
- [ ] Train on AE latent features (A2)
- [ ] Train on pooled features (B2)
- [ ] Use identical protocol as MLP experiments

**Validation Checks**

- [ ] Training stability
- [ ] No dimension mismatches
- [ ] Convergence achieved

**Outputs**

- [ ] Trained FT-Transformer models for A2 and B2

---

## 9. Test-Set Evaluation

**Targets**

Evaluate all four conditions on the identical held-out test set.

**Per-Condition Outputs**

- [ ] Confusion matrix
- [ ] Normalized confusion matrix
- [ ] Weighted F1 score
- [ ] Optional: precision/recall per class

**Consistency Checks**

- [ ] Same metrics computed for all models
- [ ] Same test data used
- [ ] No post-hoc tuning on test set

---

## 10. 2 × 2 Ablation Analysis

**Reduction Effect**

- [ ] Compare (A1 + A2) vs (B1 + B2)
- [ ] Determine AE vs Pooling benefit

**Classifier Effect**

- [ ] Compare (A1 + B1) vs (A2 + B2)
- [ ] Determine MLP vs FT-Transformer benefit

**Interaction Effect**

- [ ] Assess whether one classifier benefits more from one reduction method

**Outputs**

- [ ] Performance comparison table
- [ ] Clear ranking of the four conditions
- [ ] Interpretation of observed differences

---

## 11. External Benchmark Comparison

**Targets**

- [ ] Compare results on the 4-class subset to published baselines
- [ ] Evaluate relative performance
- [ ] Identify strengths and weaknesses

**Outputs**

- [ ] Benchmark comparison table
- [ ] Contextual interpretation

---

## 12. Final Validation

**Targets**

- [ ] Confirm conclusions follow from data
- [ ] Verify pipeline consistency across runs
- [ ] Ensure results attributable only to studied factors

---

## Quick reference — 2 × 2 conditions

|                    | MLP           | FT-Transformer |
|--------------------|---------------|----------------|
| **Autoencoder**    | A1            | A2             |
| **Structured pool**| B1            | B2             |

**Notebooks:** `00` data inspection → `01` preprocessing/splits → `02` MultiROCKET → `03` autoencoder → `04` pooling → `05` MLP → `06` FT-Transformer.  
**Train CLI:** `python -m src.training.train --config experiments/<condition>/config.yaml`
