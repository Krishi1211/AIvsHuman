# Humans vs AI: Revised Study Design

## Research Focus
This project studies how much newly added code in open source software appears AI-assisted, while treating detector outputs as probabilistic signals rather than proof of authorship.

The study uses a hybrid design:
- `AIDev` provides a known-label anchor for AI-authored and human-authored pull requests.
- A calibrated `GPTSniffer`-style detector is applied only after language, file type, and snippet-length validation.
- Multi-repository trend analysis is reported at the commit and diff-hunk level with aggressive filtering and robustness checks.

## Final Research Questions
### RQ1: Detector Validity
How well does the detector distinguish known AI-authored from human-authored code additions on a labeled benchmark, and how do errors vary by language, file category, and snippet length?

### RQ2: OSS Adoption Trends
After calibration and filtering, how does the fraction of commits with AI-like added code change over time across multiple OSS repositories?

### RQ3: Robustness and Commit Context
Are observed trends robust across thresholds, repositories, and filtering choices, and what commit-level characteristics are associated with AI-like additions?

## Repository Set
Primary repositories:
- `apache/airflow`
- `django/django`
- `elastic/elasticsearch`
- `spring-projects/spring-boot`

Fallback three-repository configuration:
- `apache/airflow`
- `django/django`
- `elastic/elasticsearch`

## Units of Analysis
- Labeled calibration unit: contiguous added-code hunks extracted from diff additions.
- OSS outcome unit: commit-level indicators aggregated from hunk-level detector scores.

## Main Outcomes
- Fraction of commits with at least one AI-like added hunk.
- Fraction of added lines that belong to AI-like hunks.
- Maximum hunk score per commit.

## Filtering Rules
The pipeline excludes or flags:
- merge commits
- bot-authored commits
- vendored or generated files
- dependency lockfiles
- docs-only changes
- config-only changes
- tiny hunks below the minimum detector length

## Calibration Strategy
Positive class:
- AI-authored pull requests from `data/aidev/all_pull_request.parquet`

Negative class:
- Human-authored pull requests from `data/aidev/human_pull_request.parquet`

Calibration is stratified by:
- language
- file category
- snippet length bucket

The detector is considered usable only in strata where calibration metrics are acceptable. Weak strata are excluded from downstream claims.

## Interpretation Rules
- Detector outputs are evidence of AI-like code, not proof of AI authorship.
- Commit metrics are contextual variables, not substitute labels.
- Commit-message detection is background material only because it is too weak for primary inference.

## Expected Outputs
- Calibration dataset and stratum-level validation summary.
- Unified filtered commit dataset for each OSS repository.
- Hunk-level detector predictions.
- Commit-level adoption metrics over time.
- Robustness tables and plots.

## Paper Structure
1. Problem statement and threat model.
2. Validation first: detector calibration on known-label data.
3. Multi-repository OSS trend analysis.
4. Robustness analysis.
5. Limitations and threats to validity.
