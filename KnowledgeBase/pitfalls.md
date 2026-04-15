# Pitfalls

## 2026-04-14

### Do not rely on background training or inference
- Foreground runs are easier to debug in this repo.
- Background runs made it harder to tell whether a job finished, crashed, or was terminated externally.

### Windows encoding can terminate long runs
- `cp950` output failed when log strings contained `R²`.
- Keep console messages ASCII-safe when possible, especially in long-running scripts.

### Pool and global model features must stay aligned
- After expanding pool features, inference broke because training and inference assembled global features differently.
- Use a single canonical feature list and avoid manual `concat` logic that can duplicate columns.

### Negative results worth remembering
- `prc2` did not recover with:
  - direct `Ridge` LOSO baseline
  - direct `HistGradientBoostingRegressor` LOSO baseline
  - disabling smoothing
  - segment length `450`
  - same-subject cross-device allowance
  - simple OOF calibration on top of the current predictions
- Relevant artifacts:
  - [Sandbox/output/prc2_c930_ridge_direct.csv](/C:/Users/yutin/code/RGB-SpO2/Sandbox/.codex/worktrees/codex-try/Sandbox/output/prc2_c930_ridge_direct.csv)
  - [Sandbox/output/prc2_c930_hgbr_direct.csv](/C:/Users/yutin/code/RGB-SpO2/Sandbox/.codex/worktrees/codex-try/Sandbox/output/prc2_c930_hgbr_direct.csv)
  - [Sandbox/output/prc2_c930_nosmooth.csv](/C:/Users/yutin/code/RGB-SpO2/Sandbox/.codex/worktrees/codex-try/Sandbox/output/prc2_c930_nosmooth.csv)
  - [Sandbox/output/prc2_segment450_experiment.csv](/C:/Users/yutin/code/RGB-SpO2/Sandbox/.codex/worktrees/codex-try/Sandbox/output/prc2_segment450_experiment.csv)
  - [Sandbox/output/prc2_c930_cross_device_allowed.csv](/C:/Users/yutin/code/RGB-SpO2/Sandbox/.codex/worktrees/codex-try/Sandbox/output/prc2_c930_cross_device_allowed.csv)
  - [Sandbox/output/prc2_c930_calibrated_subject_metrics.csv](/C:/Users/yutin/code/RGB-SpO2/Sandbox/.codex/worktrees/codex-try/Sandbox/output/prc2_c930_calibrated_subject_metrics.csv)
