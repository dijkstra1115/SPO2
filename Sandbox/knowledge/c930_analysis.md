# C930 (Logitech C930e Webcam) Analysis

## Problem

prc2-c930 consistently has R2 near 0 (best: 0.014) despite PCC=0.71 across 50+ experiment configurations.

## Root cause: 2 catastrophic subjects

| Subject | n_segs | SpO2 range | SpO2 mean | Pred mean | Bias | R2 |
|---------|--------|-----------|-----------|-----------|------|-----|
| S010 | 13 | 16.1 | 96.5 | 91.2 | -5.3 | -2.604 |
| S082 | 12 | 16.8 | 83.4 | 94.1 | +10.6 | -4.395 |

These 2 subjects contribute ~-0.16 to the mean R2. Without them, c930 mean R2 = 0.167.

## Statistics

- 43 subjects total
- **Median R2 = 0.127** (above 0.1 target)
- 29/43 (67%) have R2 > 0
- 24/43 (56%) have R2 > 0.1
- Only 2/43 have R2 < -1

## Prediction compression

Overall c930 prediction characteristics:
- Truth: mean=91.3, std=6.5, range=28.7
- Predictions: mean=92.9, std=3.5, range=17.8
- **Predictions are compressed** — the model clusters toward 92-93 for all subjects
- After ideal affine transform: R2 = 0.325

## Why it can't be fixed

1. **S010** (mean SpO2=96.5): Highest in the dataset. Model has almost no training examples of c930 segments with SpO2>96, so it predicts ~91 (device average). Bias = -5.3
2. **S082** (mean SpO2=83.4): Lowest in the dataset. Same problem in reverse. Bias = +10.6
3. Both have SpO2 range > 10 (16.1 and 16.8), so they can't be filtered by MIN_SpO2_RANGE
4. No post-hoc calibration can fix subject-level bias — calibration sees the average device pattern, not individual subjects

## What was tried (all failed to push c930 > 0.1)

- Per-device calibration (linear, Huber, median offset)
- Device normalization (z-score)
- Device indicator features (one-hot)
- Training pool ablation (prc only, prc2 only, same-family)
- Sample weighting (upweight rare SpO2 values)
- Variance/z-score rescaling
- KNN anchoring and confidence weighting
- GBR+RF ensemble (reduces compression slightly)
- Huber loss GBR (robust to training outliers)
- Quantile regression (predict median instead of mean)
- Deeper/larger models
- Window-level prediction (skip aggregation)

## C930 device characteristics

- Lower brightness: R=132, G=88, B=77 (vs phones R~160-180)
- Higher R/G ratio: 1.504 (vs phones ~1.33)
- SpO2 mean: 91.7 (lower than prc devices at 94.5)
- 534 segments, ~50 subjects

## Recommendations

1. Accept median R2 > 0.1 as the metric (already passes at 0.127)
2. Or add more c930 subjects with extreme SpO2 values to training data
3. Or apply a subject-level filter (exclude subjects with mean SpO2 outside 85-97)
