# Ensemble Methods

## Finding

GBR + RandomForest ensemble with median offset outperforms single GBR by reducing prediction range compression.

## Best ensemble config (v12-v13)

```
GBR: Huber loss, n=200, depth=4, lr=0.05, subsample=0.8
RF:  n=200-500, depth=6-8, min_leaf=3-5
Blend: 50/50 GBR/RF + per-device median offset
```

## Results comparison

| Config | c920 | c930 | i16 | i16m | ALL |
|--------|------|------|-----|------|-----|
| GBR only (devnorm+offset) | 0.262 | 0.014 | 0.169 | 0.090 | 0.250 |
| GBR+RF ensemble+offset | 0.249 | 0.013 | 0.195 | 0.123 | 0.266 |
| GBR+large RF+offset | 0.248 | 0.007 | 0.207 | 0.133 | 0.270 |

## Why ensemble works

1. **GBR compresses predictions** toward the mean (boosting bias). RF has less compression due to bagging
2. **Different error patterns**: GBR and RF make different mistakes, averaging reduces variance
3. c930 prediction std=3.5 vs truth std=6.5 — GBR alone predicts too conservatively. RF helps expand the range slightly

## Pitfalls

- **Blend ratio doesn't matter much for c930**: Tested 50/50, 40/60, 30/70, 20/80, 0/100 — c930 stuck at 0.007-0.013 regardless
- **KNN as third model is catastrophic**: 3-model ensemble with KNN produced R2=-2.0 (likely NaN propagation issue with cross-device KNN normalization)
- **Larger RF helps overall but not c930**: n=500,d=8 RF improves i16/i16m but c930 actually drops slightly
- More RF weight marginally hurts phone datasets (i15, i15m) while not helping c930

## Median offset detail

```python
y_train_pred = ensemble.predict(X_train)
dev_mask = dev_train == test_dev
offset = np.median(y_train[dev_mask] - y_train_pred[dev_mask])
y_pred = y_pred_raw + offset
```

Median offset is more robust than mean offset or linear calibration. Linear calibration (v4 Exp 1, v6 Huber regression) consistently made c930 worse by overfitting to the few same-device training samples.
