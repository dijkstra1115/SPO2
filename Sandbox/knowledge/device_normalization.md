# Device Normalization

## Finding

Per-device z-score normalization of features is the single most effective technique for cross-device generalization.

## Method

Inside the LOSO CV loop, for each device in the training set:
1. Compute per-feature mean and std from training samples of that device
2. Z-score normalize: `X = (X - mean) / std`
3. Apply the same device's stats to test data if the test subject belongs to that device

```python
for dev in np.unique(dev_train):
    dmask = dev_train == dev
    dm = np.nanmean(X_train[dmask], axis=0)
    ds = np.nanstd(X_train[dmask], axis=0)
    ds[ds < 1e-10] = 1.0
    X_train[dmask] = (X_train[dmask] - dm) / ds
    if dev == test_dev:
        X_test = (X_test - dm) / ds
```

## Impact (v4 Exp 2 vs Exp 0A baseline)

| Dataset | Without devnorm | With devnorm | Delta |
|---------|----------------|--------------|-------|
| c920 | 0.051 | 0.217 | +0.166 |
| i15 | 0.471 | 0.486 | +0.015 |
| i15m | 0.389 | 0.412 | +0.023 |
| c930 | -0.029 | -0.004 | +0.025 |
| i16 | 0.147 | 0.184 | +0.037 |
| i16m | 0.111 | 0.092 | -0.019 |
| **ALL** | **0.215** | **0.233** | **+0.018** |

Biggest impact on c920 (+0.166) because webcam features have fundamentally different distributions (R/G ratio ~1.5 vs ~1.3 for phones, lower brightness).

## Why it works

Different camera devices produce different absolute AC/DC ratios due to sensor characteristics, white balance, and illumination. Without normalization, a model trained mostly on phone data treats webcam feature values as outliers. Z-scoring removes device-specific scale/offset from features.

## Pitfall

- Must normalize **inside the CV loop** to prevent data leakage
- Device indicator features (one-hot) don't substitute for normalization — they help GBR route decisions but don't fix the feature distribution shift (v7 experiments)
- Don't normalize device indicator columns themselves — only raw feature aggregations
