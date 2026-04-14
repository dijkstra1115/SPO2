# Calibration Pitfalls

## Finding

Post-hoc per-device calibration consistently fails for devices with few training samples (especially c930). Only the simplest correction (median offset) is safe.

## What was tried and failed

| Method | c930 R2 | Problem |
|--------|---------|---------|
| No calibration | -0.004 | Baseline |
| Linear regression | -0.119 | Overfits to few same-device samples |
| Huber regression | -0.067 | Same issue, slightly less severe |
| Cross-gen family (webcam->webcam) | -0.057 | c920 and c930 are too different despite both being webcams |
| All-data calibration | -0.055 | Loses device-specific information |
| Quantile matching | -1.002 | Catastrophic — forces prediction distribution to match training, which is wrong for LOSO |
| Variance rescaling | -0.301 | Amplifies noise along with signal |
| Z-score rescaling | -0.301 | Same as variance rescaling |
| Median offset | **+0.014** | Only method that helps (barely) |

## Why calibration fails

1. **Too few same-device samples**: c930 has ~500 segments from ~50 subjects. When one subject is left out, only ~490 remain for calibration — but many subjects have similar SpO2, so the calibration data lacks diversity
2. **Linear calibration (slope + intercept) is underdetermined**: With compressed predictions (std=3.5 vs truth std=6.5), a linear fit amplifies the compression error
3. **Device families aren't close enough**: c920 (Logitech C920) and c930 (Logitech C930) have different enough sensor characteristics that cross-calibration doesn't help

## Safe approach: median offset only

Median offset shifts predictions by a constant (the median residual on same-device training data). It's robust because:
- Only estimates 1 parameter (offset), not 2 (slope + intercept)
- Median is resistant to outliers
- Doesn't try to change prediction scale/variance

## Lesson

For devices with limited training data, simpler corrections are always better. Any calibration method that estimates more than 1 parameter will overfit. If you must calibrate, check that `dev_mask.sum()` is large enough relative to the number of parameters being estimated.
