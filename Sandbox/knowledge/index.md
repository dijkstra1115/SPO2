# Knowledge Base: rPPG SpO2 Prediction Experiments

This folder records valuable findings, pitfalls, and lessons learned from SpO2 prediction experiments. Each topic is a separate `.md` file linked below.

## Findings

- [Segment Length](segment_length.md) — seg=600 is the sweet spot; shorter segments increase noise, longer ones lose subjects
- [Device Normalization](device_normalization.md) — Per-device z-score normalization is essential for cross-device generalization
- [Ensemble Methods](ensemble_methods.md) — GBR+RF ensemble outperforms single models by reducing prediction compression
- [Calibration Pitfalls](calibration_pitfalls.md) — Post-hoc calibration consistently fails for low-sample devices
- [C930 Analysis](c930_analysis.md) — Why prc2-c930 R2 is stuck near 0 despite decent PCC
- [R2 vs PCC Gap](r2_vs_pcc_gap.md) — Root cause of "good correlation, bad R2" and what works/doesn't

## Best Configurations

- [Best Results Summary](best_results.md) — Top configurations across all experiments (v4-v13)

## Experiment Log

- [Experiment Index](experiment_index.md) — Quick reference of all experiments and their key outcomes
