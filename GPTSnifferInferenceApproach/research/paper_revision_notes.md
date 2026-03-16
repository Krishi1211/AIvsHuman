# Paper Revision Notes

## New Core Story
The paper should now be framed as a calibrated detector study rather than a raw detector study.

Recommended one-sentence framing:

> We calibrate a GPTSniffer-style CodeBERT detector on labeled AI and human code additions, then use the calibrated detector to estimate how often AI-like code additions appear in sampled OSS commits over time across multiple repositories.

## Final RQs
### RQ1: Detector Validity
How well does the calibrated detector distinguish known AI-authored from human-authored code additions, and how does performance vary by language and artifact type?

### RQ2: OSS Adoption Trends
How does the fraction of sampled commits with AI-like added code evolve over time across `airflow`, `django`, and `elasticsearch`?

### RQ3: Robustness and Context
How robust are the observed trends to threshold choice, and how do results differ by repository?

## What You Can Claim Safely
- The detector achieved `balanced accuracy = 0.836` and `ROC AUC = 0.916` on a held-out calibration sample.
- The selected operating threshold was `0.45`.
- The downstream OSS analysis was restricted to `Python/Java`, `source/test`, and non-`very_short` added hunks.
- The empirical repository study covered `airflow`, `django`, and `elasticsearch`.
- The pooled mean AI-like commit fraction rose modestly from about `0.398` before 2023 to about `0.417` in 2023 and later.
- A better-supported pre/post comparison is `2021-2022` versus `2023-2025`, where the medium-threshold weighted AI-like commit fraction rises from about `0.412` to about `0.423`.
- The pre-2020 window is too sparse to support a strong standalone baseline claim.
- Repository means differed, with `elasticsearch` highest on average among the three analyzed repositories.

## What You Should Not Claim
- Do not claim proof of AI authorship.
- Do not claim that large commits are inherently AI-generated.
- Do not use commit-message detection as primary evidence.
- Do not present the old insertions-to-complexity proxy as a detector substitute.

## Methods Section Changes
Replace the old methodology emphasis with:
1. AIDev-based positive class plus human-control PR patch sampling for the negative class.
2. Hunk-level calibration before repository-scale inference.
3. Aggressive filtering:
   - merge commits removed
   - bot identities removed
   - vendored/generated files removed
   - lockfiles removed
   - docs/config excluded from main analysis
   - very short hunks excluded from main analysis
4. Monthly sampled-commit inference for tractable multi-repo analysis.

## Results Section Structure
### Section 1: Detector Calibration
Use:
- `research/assets/figures/calibration_by_language.png`
- `research/assets/figures/calibration_by_file_category.png`
- `research/assets/tables/calibration_by_language.csv`
- `research/assets/tables/calibration_by_file_category.csv`

### Section 2: Repository Trend Results
Use:
- `research/assets/figures/pooled_monthly_trend.png`
- `research/assets/figures/repo_monthly_trends.png`
- `research/assets/figures/pre_post_llm_trend.png`
- `research/assets/tables/pooled_monthly_summary_medium.csv`
- `research/assets/tables/repo_monthly_summary_medium.csv`
- `research/assets/tables/pre_post_llm_period_summary.csv`

### Section 3: Robustness
Use:
- `research/assets/figures/threshold_robustness.png`
- `research/assets/tables/threshold_robustness.csv`

### Section 4: Sampling and Exclusions
Use:
- `research/assets/tables/exclusions_and_thresholds.csv`
- `research/assets/tables/repo_sample_sizes.csv`

## Discussion Section Guidance
Your interpretation should be:
- The calibrated detector finds non-trivial AI-like code signals in all three repositories.
- The post-2023 increase is present but modest, not explosive.
- The cleanest temporal comparison is `2021-2022` versus `2023-2025`, not pre-2020 versus post-2020, because the earliest years are thinly sampled.
- Cross-repository heterogeneity matters.
- The study is strongest as evidence of changing AI-like code signals under calibration and filtering, not absolute AI usage rates.
- Extra fine-tuning, larger calibration sets, and a stronger human negative set would likely make the pre/post distinction clearer.

## Limitations to State Explicitly
- Human negative examples were constructed from sampled human-control PR patches, not a full mirrored file-level dataset.
- The repository-scale trend analysis used sampled commits per month for computational tractability.
- Detector probabilities may still be affected by language, test-code prevalence, and repository conventions.
- Results are limited to the calibrated strata and selected repositories.

## Deliverables to Cite in the Draft
- Design note: `research/revised_study_design.md`
- Calibration summary: `research/assets/draft_results_summary.md`
- Detector outputs: `data/calibration_results`
- Repo analysis outputs: `data/analysis`
