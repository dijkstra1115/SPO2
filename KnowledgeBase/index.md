# Experiment Knowledge Base

## Purpose
This folder stores reusable findings from experiments, debugging sessions, and failed attempts. Read this index before starting a new training or inference cycle, and update it after any result that changes the current understanding of the project.

## Documents
- [Findings](./findings.md): validated results, current best settings, and metric snapshots worth preserving.
- [Pitfalls](./pitfalls.md): repeated failure modes, invalid assumptions, and debugging notes that should not be rediscovered.
- [PRC2 Recovery Plan](./prc2_recovery_plan.md): focused plan for getting `prc2-*` above the target threshold.

## Update Rules
- Add a note when an experiment changes the best known result, disproves an approach, or exposes a reproducible bug.
- Prefer short entries with date, scope, conclusion, and artifact paths.
- Reference concrete files under `Sandbox/output/` whenever possible.
- Update this index if a new topic file is added.

## Current Focus
- `prc-*` currently meets the `R2 > 0.1` target with a domain-specialized raw pool.
- `prc2-*` still fails the target; current evidence suggests the bottleneck is feature representation, not simple pool pruning.
