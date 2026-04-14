# R2 vs PCC Gap: Good Correlation, Bad R2

## Problem

Many configurations achieve PCC 0.6-0.7 but R2 near 0 or negative. This is the core challenge for c930 and was the original problem for all prc2 datasets.

## What it means

- **PCC (Pearson Correlation)**: Measures if predictions follow the same trend as truth (shape). Invariant to scale and offset.
- **R2 (Coefficient of Determination)**: Measures if predictions match truth in absolute value. Sensitive to bias (offset) and scale.

A model with PCC=0.7 and R2=0 predicts the **right shape** but at the **wrong level**.

## Root causes

### 1. Ensemble bias (model pool approach)
Each linear model's intercept encodes its training segment's mean SpO2. When averaging models with different biases, the result anchors to an arbitrary mean. This was the original problem with the model pool + similarity ensemble approach.

**Fix**: GBR global model doesn't have this issue (single model, no bias averaging).

### 2. GBR prediction compression
GBR regularizes toward the mean, producing predictions with lower variance than truth (std=3.5 vs 6.5 for c930). This creates a systematic R2 penalty for subjects with extreme mean SpO2.

**Fix**: GBR+RF ensemble slightly reduces compression. But can't fully eliminate it.

### 3. Device-specific offset
Different devices have different mean SpO2 in the training data. A model trained on phone-dominated data predicts toward phone-average SpO2 for webcam subjects.

**Fix**: Device normalization + median offset mostly solves this for phones (i16m improved from 0.033 to 0.113). Less effective for c930 (too few samples for reliable offset).

## Key insight from early experiments

GBR blend weight was the most impactful parameter in early model pool experiments:
- Higher GBR blend (0.35-0.5): better PCC but worse R2 (GBR adds scale distortion)
- Lower GBR blend (0.1-0.2): better R2 but reduced PCC

This confirmed that the R2 problem was about scale/offset, not about the model's ability to capture SpO2 variation patterns.

## Negative R2 is driven by outliers

Negative R2 doesn't mean the model is useless — it means a few catastrophic subjects dominate the metric. For prc2 datasets:
- Excluding ~7 worst subjects: c930 R2 went from -0.180 to -0.015, i16 from -0.315 to -0.009, i16m from -0.199 to +0.113
- c930 median R2 = 0.127 even when mean R2 = 0.013
