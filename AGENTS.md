# Repository Guidelines

## Project Structure & Module Organization
`Sandbox/` contains the core training and inference workflow for SpO2 prediction. Use [Sandbox/common.py](/C:/Users/yutin/code/RGB-SpO2/Sandbox/.codex/worktrees/codex-try/Sandbox/common.py) for shared feature extraction, filtering, and evaluation logic, with [Sandbox/train.py](/C:/Users/yutin/code/RGB-SpO2/Sandbox/.codex/worktrees/codex-try/Sandbox/train.py) and [Sandbox/infer.py](/C:/Users/yutin/code/RGB-SpO2/Sandbox/.codex/worktrees/codex-try/Sandbox/infer.py) as entry points. `Sandbox/data_new/` stores sample CSV inputs; `Sandbox/output/` stores generated metrics and model artifacts. `SQI/` holds standalone signal-quality analysis scripts and exported plots. `Data Preparation/` contains one-off dataset preparation utilities.

`KnowledgeBase/` stores experiment memory for future contributors. Start at [KnowledgeBase/index.md](/C:/Users/yutin/code/RGB-SpO2/Sandbox/.codex/worktrees/codex-try/KnowledgeBase/index.md), then follow topic files such as findings or pitfalls before repeating old experiments.

## Build, Test, and Development Commands
Use the shared Conda environment for all work in this repository:

```powershell
conda activate RGB-SpO2
pip install -r requirements.txt
```

Run model training from `Sandbox/`:

```powershell
python train.py
python infer.py
```

Run SQI analysis from `SQI/`:

```powershell
python analysis.py
```

Before running, verify the CSV lists in `train.py` and `infer.py` match the available files under `Sandbox/data_new/` or your local dataset mount.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, snake_case for functions/variables, UPPER_CASE for constants such as `SEGMENT_LENGTH` and `TOP_K`. Keep new logic in `common.py` when it is shared by training and inference. Prefer small script-level changes over introducing new frameworks. No formatter or linter is configured, so keep imports tidy and match surrounding style.

Because several scripts and comments include Chinese text, always read and write source files as UTF-8. Do not convert files to ANSI, Big5, or UTF-8 with BOM unless a file already uses that encoding. When adding file I/O, specify `encoding="utf-8"` unless there is a documented reason not to.

## Testing Guidelines
There is no formal `tests/` suite yet. Treat reproducible script runs as the baseline:

```powershell
cd Sandbox
python train.py
python infer.py
```

Check that updated runs regenerate expected files in `Sandbox/output/` and review summary CSVs for metric regressions. For SQI changes, rerun `SQI/analysis.py` and inspect the generated PNGs.

Run training and inference in the foreground unless there is a strong reason not to. This repository benefits from direct visibility into stdout, tracebacks, and generated summaries, and foreground runs avoid extra debugging around background process state or Windows encoding issues.

After any meaningful experiment, update `KnowledgeBase/` with:
- the conclusion
- the exact config or constraint tested
- the output files that support the conclusion

Do not leave important findings only in chat history or ad hoc CSV filenames.

## Commit & Pull Request Guidelines
Recent history uses short, imperative summaries, often with tags such as `[Feature]` or `[Experiment]`. Keep that pattern, for example: `[Experiment] Tune TOP_K for c920 dataset`. PRs should state the dataset/configuration changed, expected metric impact, and any regenerated outputs. Include plots or CSV excerpts when behavior changes materially.
