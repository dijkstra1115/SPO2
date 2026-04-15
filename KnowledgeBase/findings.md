# Findings

## 2026-04-14

### `prc-*` can pass with domain-specialized raw pool
- Best confirmed configuration so far: raw pool specialized by dataset prefix, `top_k=20`, `segment_length=900`.
- Result snapshot from [Sandbox/output/infer_summary_topk20_seg900.csv](/C:/Users/yutin/code/RGB-SpO2/Sandbox/.codex/worktrees/codex-try/Sandbox/output/infer_summary_topk20_seg900.csv):
  - `prc-c920`: `R2 = 0.2845`
  - `prc-i15`: `R2 = 0.2462`
  - `prc-i15m`: `R2 = 0.1794`
- Conclusion: for `prc-*`, the main gain came from pool specialization rather than adding a larger generic ensemble.

### `prc2-*` remains below target after pool-only tuning
- Latest `prc2` summary from [Sandbox/output/infer_summary_topk20_seg900.csv](/C:/Users/yutin/code/RGB-SpO2/Sandbox/.codex/worktrees/codex-try/Sandbox/output/infer_summary_topk20_seg900.csv):
  - `prc2-c930`: `R2 = -0.1689`
  - `prc2-i16`: `R2 = -0.3832`
  - `prc2-i16m`: `R2 = -0.6605`
- Conclusion: `prc2-*` is not fixed by simple raw-pool specialization or `top_k` tuning.

### Feature shift exists between `prc` and `prc2`
- Reference: [Sandbox/output/prc_vs_prc2_feature_shift.csv](/C:/Users/yutin/code/RGB-SpO2/Sandbox/.codex/worktrees/codex-try/Sandbox/output/prc_vs_prc2_feature_shift.csv)
- Largest observed shifts were in `R_mean`, `B_acdc`, `RoR_RB_acdc`, `G_acdc`, and `RoR_RG_acdc`.
- Conclusion: cross-domain degradation is likely tied to feature distribution shift, so future work should prioritize feature-layer redesign over more pool pruning.
