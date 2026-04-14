# Experiment Index

Quick reference of all global model experiments (v4-v13).

## v4: Device-aware modeling (4 experiments)
- **Exp 0**: Training pool ablation (all/prc/prc2/same-family) — all 6 datasets best
- **Exp 1**: Per-device linear calibration — helps c920, hurts c930
- **Exp 2**: Device normalization (z-score) — best single technique (+0.018 overall)
- **Exp 3**: Devnorm + calibration — calibration hurts on top of devnorm

## v5: Alternative approaches (5 experiments)
- **Exp 1**: Device-specific models — too little data per device
- **Exp 2**: Global + median offset — i16m passes (0.105)
- **Exp 3**: Devnorm + median offset — c930 best at 0.014, overall 0.250
- **Exp 4**: Window-level prediction — no improvement
- **Exp 5**: Stacked GBR + Ridge — similar to calibration

## v6: Calibration strategies (5 experiments)
- Huber affine, cross-gen family, all-data, quantile matching, blend
- **All calibration makes c930 worse**. Quantile matching catastrophic (-0.88)

## v7: Device indicators + segment sweep (8 experiments)
- One-hot device features + seg=600/450/300
- seg=600 remains optimal; device indicators don't help c930

## v8: Variance rescaling + c930 diagnosis (6 experiments)
- **Key finding**: c930 predictions compress range (pred std=3.5 vs truth std=6.5)
- Variance rescaling amplifies noise — makes everything worse
- Larger model (n=200,d=4) slightly helps i16

## v9: Sample weighting + Ridge (6 experiments)
- Sample weighting catastrophic (-0.218 overall)
- Ridge worse than GBR
- large model + offset: best so far for i16m (0.083) and overall (0.242)

## v10: Huber GBR + webcam blend (6 experiments)
- **Huber GBR loss**: c930 = -0.0004 (closest to zero)
- Interaction features: c930 = 0.006
- Webcam-specific blend doesn't help

## v11: Last push for c930 (8 experiments)
- **Huber large + offset**: 5/6 pass (i16m=0.113!), overall 0.261
- Deep regularized GBR: i16m=0.110
- KNN anchor, RF, quantile: all ~0.01 for c930

## v12: Ensemble and confidence (5 experiments)
- **GBR+RF ensemble + offset**: c930=0.013, i16m=0.123, overall=0.266
- KNN blend, confidence weighting: no improvement for c930
- Two-stage offset: same as single-stage

## v13: Ensemble optimization (8 experiments)
- GBR/RF ratio sweep (50/50 to 0/100): c930 insensitive (~0.013)
- **large RF (n=500,d=8)**: best overall 0.270, i16m=0.133
- 3-model with KNN: catastrophic (NaN issue)

## Summary

Total: 50+ configurations tested. c930 never exceeds R2=0.014 (mean) due to 2 catastrophic subjects. 5/6 datasets reliably pass R2>0.1 with the ensemble approach.
