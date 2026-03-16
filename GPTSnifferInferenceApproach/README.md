# Who Wrote It: Humans or AI?

This folder contains the final reproducible pipeline for studying AI-like code signals in open-source software using a calibrated GPTSniffer-style detector.

The final study uses:
- `AIDev` as the calibration/ground-truth anchor
- a `CodeBERT` + logistic-regression detector as a lightweight GPTSniffer-style classifier
- 3 OSS repositories for the final analysis:
  - `apache/airflow`
  - `django/django`
  - `elastic/elasticsearch`

The workflow was cleaned so an external user can follow this README and reproduce the main outputs from scratch.

## What Is In This Repo

- `download_aidev_dataset.py`
  - Downloads the AIDev dataset from Hugging Face.
- `research_pipeline/`
  - Final code used for calibration, OSS mining, inference, and asset generation.
- `research/`
  - Final study notes, paper rewrite guidance, and generated figures/tables.
- `data/`
  - Downloaded dataset and derived outputs.
- `../GPTSniffer/`
  - Reference GPTSniffer codebase kept at the repo root for context.

## Final Methodology

The final methodology follows a hybrid design:

1. Use AIDev as the known-label calibration anchor.
2. Build a hunk-level labeled dataset from AI-authored and human-authored code additions.
3. Calibrate a GPTSniffer-style detector before using it on OSS repositories.
4. Restrict repository-scale claims to filtered commit/diff data and calibrated strata.
5. Report results as commit-level AI-like adoption trends, not as proof of AI authorship.

### Research Questions

The final study is organized around 3 RQs:

1. `RQ1: Detector Validity`
   - How well does the calibrated detector distinguish known AI-authored from human-authored added-code hunks?

2. `RQ2: OSS Adoption Trend`
   - After calibration and filtering, how does the fraction of commits with AI-like added code change over time across multiple OSS repositories?

3. `RQ3: Robustness and Commit Context`
   - How stable are the trends under threshold changes and repository differences?

### Main Filtering Rules

The final repository-scale analysis excludes or de-emphasizes:
- merge commits
- bot-authored commits
- vendored/generated files
- lockfiles
- docs/config from the main analysis
- very short hunks from the main analysis

### Important Interpretation Rule

All detector outputs are treated as **AI-like signals**, not definitive proof of AI authorship.

## Datasets Used

### 1. AIDev Dataset

Downloaded into `data/aidev`.

Main files used:
- `data/aidev/all_pull_request.parquet`
- `data/aidev/human_pull_request.parquet`
- `data/aidev/pr_commit_details.parquet`
- `data/aidev/all_repository.parquet`

How it was used:
- AI PRs from `all_pull_request.parquet` were used as the positive class anchor.
- Human PRs from `human_pull_request.parquet` were used as the negative class anchor.
- AI-side file patches came from `pr_commit_details.parquet`.
- Human-side patch content was reconstructed by fetching sampled human PR `.patch` files from GitHub.

### 2. OSS Repositories

The final analysis used:
- `apache/airflow`
- `django/django`
- `elastic/elasticsearch`

Repository metadata and extracted outputs live in:
- `repos/`
- `data/oss/`

## Final Results Summary

### Calibration Performance

Saved in:
- `data/calibration_results/overall_metrics.json`
- `data/calibration_results/frozen_config.json`
- `data/calibration_results/metrics_by_language.csv`
- `data/calibration_results/metrics_by_file_category.csv`

Key held-out calibration numbers:
- Balanced accuracy: `0.836`
- ROC AUC: `0.916`
- Precision: `0.914`
- Recall: `0.876`
- Selected threshold: `0.45`

### Repository Mining Coverage

Saved in:
- `data/oss/airflow/summary.json`
- `data/oss/django/summary.json`
- `data/oss/elasticsearch/summary.json`

Mined commit/hunk counts:
- `airflow`: `23,556` commits, `518,811` hunks
- `django`: `5,014` commits, `108,454` hunks
- `elasticsearch`: `34,897` commits, `974,953` hunks

### Final Repo-Scale Analysis

Saved in:
- `data/analysis/airflow/`
- `data/analysis/django/`
- `data/analysis/elasticsearch/`
- `data/analysis/all_commit_analysis.parquet`
- `data/analysis/all_monthly_summary.csv`
- `data/analysis/pooled_monthly_summary.csv`

The final inference used a monthly sampled-commit design for tractability.

Per-repo medium-threshold averages:
- `airflow`: commit AI-like fraction `0.4000`
- `django`: commit AI-like fraction `0.3985`
- `elasticsearch`: commit AI-like fraction `0.4542`

Pooled medium-threshold trend:
- Pre-2023 mean: about `0.398`
- 2023+ mean: about `0.417`
- Better-supported period comparison: `2021-2022 = 0.412` vs `2023-2025 = 0.423`
- Important caveat: pre-2020 observations are too sparse to support a strong standalone baseline claim.

## Figures and Tables

Generated assets live in `research/assets`.

### Figures

- `research/assets/figures/calibration_by_language.png`
- `research/assets/figures/calibration_by_file_category.png`
- `research/assets/figures/pooled_monthly_trend.png`
- `research/assets/figures/repo_monthly_trends.png`
- `research/assets/figures/pre_post_llm_trend.png`
- `research/assets/figures/threshold_robustness.png`

### Tables

- `research/assets/tables/calibration_by_language.csv`
- `research/assets/tables/calibration_by_file_category.csv`
- `research/assets/tables/pooled_monthly_summary_medium.csv`
- `research/assets/tables/pre_post_llm_period_summary.csv`
- `research/assets/tables/pre_post_llm_yearly_summary.csv`
- `research/assets/tables/repo_monthly_summary_medium.csv`
- `research/assets/tables/threshold_robustness.csv`
- `research/assets/tables/exclusions_and_thresholds.csv`
- `research/assets/tables/repo_sample_sizes.csv`

### Draft Narrative Helpers

- `research/assets/draft_results_summary.md`
- `research/paper_revision_notes.md`
- `research/revised_study_design.md`

## Code Map: What Each Script Does

### Dataset Download

- `download_aidev_dataset.py`
  - Downloads AIDev from Hugging Face into `data/aidev`.

### Shared Utilities

- `research_pipeline/common.py`
  - File classification, bot filtering, diff parsing, and added-hunk extraction utilities.

### Calibration

- `research_pipeline/build_calibration_dataset.py`
  - Builds the labeled hunk-level calibration dataset from AIDev AI PRs and sampled human-control PR patches.

- `research_pipeline/gptsniffer_detector.py`
  - Implements the final detector used in this project.
  - This is a lightweight GPTSniffer-style detector using:
    - `CodeBERT` embeddings
    - a logistic regression classifier

- `research_pipeline/calibrate_gptsniffer.py`
  - Trains/calibrates the detector on the labeled hunk dataset.
  - Produces threshold selection and stratum-level metrics.

### OSS Mining

- `research_pipeline/extract_oss_hunks.py`
  - Clones/mines the OSS repositories with PyDriller.
  - Produces:
    - commit-level metrics
    - hunk-level added-code snippets

### OSS Inference and Aggregation

- `research_pipeline/run_multi_repo_analysis.py`
  - Runs the calibrated detector on sampled OSS commit hunks.
  - Produces:
    - hunk-level probabilities
    - commit-level AI-like indicators
    - monthly summaries
    - pooled summaries

### Paper Assets

- `research_pipeline/generate_paper_assets.py`
  - Builds the final plots, tables, and draft summary files.

## How To Reproduce Everything From Scratch

These are the exact steps an external user should follow.

### 1. Create a clean Python environment

```bash
python3 -m venv .venv-research
source .venv-research/bin/activate
pip install -r requirements-research.txt
```

### 2. Download AIDev

```bash
python3 download_aidev_dataset.py
```

This writes the dataset into `data/aidev`.

### 3. Build the calibration dataset

```bash
python research_pipeline/build_calibration_dataset.py
```

Expected outputs:
- `data/calibration/calibration_hunks_full.parquet`
- `data/calibration/calibration_hunks_sampled.parquet`
- summary/count CSVs in `data/calibration/`

### 4. Calibrate the detector

```bash
python research_pipeline/calibrate_gptsniffer.py
```

Expected outputs:
- `data/calibration_results/overall_metrics.json`
- `data/calibration_results/frozen_config.json`
- `data/calibration_results/threshold_metrics.csv`
- detector files under `data/calibration_results/detector/`

### 5. Mine the OSS repositories with PyDriller

```bash
python research_pipeline/extract_oss_hunks.py --repos airflow django elasticsearch
```

Expected outputs:
- `data/oss/airflow/`
- `data/oss/django/`
- `data/oss/elasticsearch/`

### 6. Run calibrated detector inference

The final run used a sampled monthly design:

```bash
python research_pipeline/run_multi_repo_analysis.py --repos airflow django elasticsearch --max-commits-per-month 20
```

Expected outputs:
- `data/analysis/airflow/`
- `data/analysis/django/`
- `data/analysis/elasticsearch/`
- `data/analysis/all_commit_analysis.parquet`
- `data/analysis/all_monthly_summary.csv`
- `data/analysis/pooled_monthly_summary.csv`

### 7. Generate figures and paper assets

```bash
python research_pipeline/generate_paper_assets.py
```

Expected outputs:
- `research/assets/figures/`
- `research/assets/tables/`
- `research/assets/draft_results_summary.md`

## Where To Change Things To Improve or Fine-Tune Results

### 1. Increase or rebalance the calibration sample

Edit:
- `research_pipeline/build_calibration_dataset.py`

Useful knobs:
- `--max-per-stratum`
- `--human-prs-per-language`
- sampling logic for human PRs

Why:
- More balanced negative examples can improve threshold stability.
- More human Java/Python patch coverage would likely improve calibration quality.

### 2. Change detector capacity or sequence length

Edit:
- `research_pipeline/gptsniffer_detector.py`

Useful knobs:
- `model_name`
- `max_length`
- `batch_size`

Why:
- Longer context may help some hunks.
- A stronger code model may separate AI/human code better.

### 3. Change calibration budget

Edit:
- `research_pipeline/calibrate_gptsniffer.py`

Useful knobs:
- `--max-total-rows`
- threshold selection logic
- stratum acceptance rules in `derive_frozen_config()`

Why:
- Larger training/evaluation sets may improve detector stability.
- Different acceptance rules can tighten or loosen downstream claims.

### 4. Change filtering rules

Edit:
- `research_pipeline/common.py`

Useful knobs:
- bot markers
- file-category rules
- generated/vendored patterns
- snippet-length handling

Why:
- Better filtering reduces false positives from noisy artifacts.

### 5. Change OSS sampling strategy

Edit:
- `research_pipeline/run_multi_repo_analysis.py`

Useful knobs:
- `--max-commits-per-month`
- the monthly sampling logic
- downstream commit aggregation

Why:
- More commits per month improves representativeness but increases runtime.
- Different sampling can change variance and robustness.

## Could The Results Be Better If Trained Somewhere Else?

Yes.

The current detector is a practical, laptop-friendly GPTSniffer-style approximation. Results could likely improve if you train or fine-tune in a stronger environment:

- On a GPU machine or cloud notebook
  - You could run larger calibration sets.
  - You could increase sequence length and batch size safely.

- With full end-to-end CodeBERT fine-tuning
  - Instead of frozen embeddings plus logistic regression.
  - This would likely improve separation if you have enough labeled data.

- With a better human negative set
  - Especially file-level human patches that mirror the AI patch granularity more closely.

- With more languages and larger labeled strata
  - Especially if you want to extend beyond Python/Java.

In short:
- `Yes, the results could be better if trained elsewhere, especially on a GPU-backed environment with a stronger fine-tuning setup and a richer human patch dataset.`

## What Was Removed During Cleanup

To keep verification clean, the following obsolete items were removed:
- the broken old virtualenv at the workspace root
- the old `.venv-pydriller`
- old temporary Elasticsearch-only PyDriller clone folders
- the incomplete `spring-boot` clone
- the unrelated old `data/elasticsearch_metrics` outputs

This repo now reflects the final reproducible workflow used for the project.

## Recommended Reading Order

For an external reviewer:

1. Read this `README.md`
2. Read `research/revised_study_design.md`
3. Read `research/assets/draft_results_summary.md`
4. Inspect `research/assets/figures/`
5. Verify calibration in `data/calibration_results/`
6. Verify repository outputs in `data/oss/` and `data/analysis/`

## Notes

- `../GPTSniffer/` is kept as reference context at the repo root, but the final implemented detector for this project is the one under `research_pipeline/gptsniffer_detector.py`.
- The final analysis used the three-repository fallback from the plan because it was sufficient and completed cleanly.
