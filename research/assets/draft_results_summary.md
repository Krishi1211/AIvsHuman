# Draft Results Summary

## RQ1: Detector Validity
- Balanced accuracy: 0.836
- ROC AUC: 0.916
- Precision: 0.914
- Recall: 0.876
- Selected threshold: 0.45

## RQ2: OSS Adoption Trends
- Earliest pooled monthly AI-like commit fraction: 0.306
- Latest pooled monthly AI-like commit fraction: 0.447
- Repositories analyzed: airflow, django, elasticsearch
- Better-supported pre/post comparison: `2021-2022 = 0.412` vs `2023-2025 = 0.423` at the medium threshold.
- Pre-2020 observations are too sparse for a strong standalone baseline claim.


## RQ3: Robustness and Commit Context
- Threshold sensitivity and per-repository heterogeneity are captured in the robustness tables and plots.
- Main claims should be restricted to allowed language/file-category/length strata from the frozen calibration config.

## Interpretation
- Results indicate AI-like code signals rather than definitive AI authorship.
- Calibration should be foregrounded before any repository-scale trend claim.
