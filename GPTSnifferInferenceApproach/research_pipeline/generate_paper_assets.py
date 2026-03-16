#!/usr/bin/env python3
"""Generate figures, tables, and a short write-up for the paper draft."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ASSETS_DIR = Path("research/assets")
FIGURES_DIR = ASSETS_DIR / "figures"
TABLES_DIR = ASSETS_DIR / "tables"


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def plot_calibration_by_language(calibration_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(calibration_dir / "metrics_by_language.csv")
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="language", y="balanced_accuracy", hue="language", legend=False)
    plt.ylim(0, 1)
    plt.title("Calibration Performance by Language")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_by_language.png", dpi=200)
    plt.close()
    df.to_csv(TABLES_DIR / "calibration_by_language.csv", index=False)
    return df


def plot_calibration_by_category(calibration_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(calibration_dir / "metrics_by_file_category.csv")
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="file_category", y="balanced_accuracy", hue="file_category", legend=False)
    plt.ylim(0, 1)
    plt.title("Calibration Performance by File Category")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_by_file_category.png", dpi=200)
    plt.close()
    df.to_csv(TABLES_DIR / "calibration_by_file_category.csv", index=False)
    return df


def plot_pooled_trend(analysis_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(analysis_dir / "pooled_monthly_summary.csv")
    medium = df[(df["threshold_label"] == "medium") & (df["calendar_month"] >= "2021-01")].copy()
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=medium, x="calendar_month", y="commit_ai_fraction")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("AI-like commit fraction")
    plt.title("Pooled Monthly AI-like Commit Fraction")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pooled_monthly_trend.png", dpi=200)
    plt.close()
    medium.to_csv(TABLES_DIR / "pooled_monthly_summary_medium.csv", index=False)
    return medium


def plot_repo_trends(analysis_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(analysis_dir / "all_monthly_summary.csv")
    medium = df[(df["threshold_label"] == "medium") & (df["calendar_month"] >= "2021-01")].copy()
    plt.figure(figsize=(11, 5))
    sns.lineplot(data=medium, x="calendar_month", y="commit_ai_fraction", hue="repo")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("AI-like commit fraction")
    plt.title("Per-Repository Monthly AI-like Commit Fraction")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "repo_monthly_trends.png", dpi=200)
    plt.close()
    medium.to_csv(TABLES_DIR / "repo_monthly_summary_medium.csv", index=False)
    return medium


def plot_threshold_robustness(analysis_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(analysis_dir / "all_monthly_summary.csv")
    robustness = (
        df.groupby(["repo", "threshold_label"])["commit_ai_fraction"]
        .mean()
        .reset_index(name="mean_commit_ai_fraction")
    )
    plt.figure(figsize=(8, 4))
    sns.barplot(data=robustness, x="repo", y="mean_commit_ai_fraction", hue="threshold_label")
    plt.xticks(rotation=20, ha="right")
    plt.title("Robustness Across Detector Thresholds")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_robustness.png", dpi=200)
    plt.close()
    robustness.to_csv(TABLES_DIR / "threshold_robustness.csv", index=False)
    return robustness


def plot_pre_post_llm_trend(analysis_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(analysis_dir / "pooled_monthly_summary.csv")
    medium = df[df["threshold_label"] == "medium"].copy()
    medium["year"] = medium["calendar_month"].str.slice(0, 4).astype(int)
    medium = medium[medium["year"] >= 2021].copy()

    annual = (
        medium.assign(weighted_ai_line_numerator=medium["mean_ai_line_fraction"] * medium["commits"])
        .groupby("year", as_index=False)
        .agg(
            commits=("commits", "sum"),
            ai_like_commits=("ai_like_commits", "sum"),
            weighted_ai_line_numerator=("weighted_ai_line_numerator", "sum"),
        )
    )
    annual["weighted_commit_ai_fraction"] = annual["ai_like_commits"] / annual["commits"]
    annual["weighted_mean_ai_line_fraction"] = annual["weighted_ai_line_numerator"] / annual["commits"]
    annual = annual.drop(columns=["weighted_ai_line_numerator"])
    annual["period"] = annual["year"].apply(
        lambda year: "2021-2022 pre-mainstream LLM" if year <= 2022 else "2023-2025 post-mainstream LLM"
    )

    period_summary = (
        annual.assign(
            weighted_ai_line_numerator=annual["weighted_mean_ai_line_fraction"] * annual["commits"],
            year_str=annual["year"].astype(str),
        )
        .groupby("period", as_index=False)
        .agg(
            years=("year_str", ",".join),
            commits=("commits", "sum"),
            ai_like_commits=("ai_like_commits", "sum"),
            weighted_ai_line_numerator=("weighted_ai_line_numerator", "sum"),
        )
    )
    period_summary["weighted_commit_ai_fraction"] = period_summary["ai_like_commits"] / period_summary["commits"]
    period_summary["weighted_mean_ai_line_fraction"] = (
        period_summary["weighted_ai_line_numerator"] / period_summary["commits"]
    )
    period_summary = period_summary.drop(columns=["weighted_ai_line_numerator"])

    plot_df = annual.copy()
    plot_df["year_label"] = plot_df["year"].astype(str)

    plt.figure(figsize=(9, 4.5))
    ax = sns.barplot(
        data=plot_df,
        x="year_label",
        y="weighted_commit_ai_fraction",
        hue="period",
        dodge=False,
        palette={
            "2021-2022 pre-mainstream LLM": "#8da0cb",
            "2023-2025 post-mainstream LLM": "#fc8d62",
        },
    )
    pre_mean = float(
        period_summary.loc[
            period_summary["period"] == "2021-2022 pre-mainstream LLM",
            "weighted_commit_ai_fraction",
        ].iloc[0]
    )
    post_mean = float(
        period_summary.loc[
            period_summary["period"] == "2023-2025 post-mainstream LLM",
            "weighted_commit_ai_fraction",
        ].iloc[0]
    )
    ax.axhline(pre_mean, color="#4c72b0", linestyle="--", linewidth=1.5, label="2021-2022 mean")
    ax.axhline(post_mean, color="#dd8452", linestyle="--", linewidth=1.5, label="2023-2025 mean")
    ax.set_ylim(0, max(plot_df["weighted_commit_ai_fraction"].max() + 0.04, 0.5))
    ax.set_ylabel("Weighted AI-like commit fraction")
    ax.set_xlabel("Calendar year")
    ax.set_title("Modest Increase in AI-like Commit Fraction After 2022")
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pre_post_llm_trend.png", dpi=200)
    plt.close()

    annual.to_csv(TABLES_DIR / "pre_post_llm_yearly_summary.csv", index=False)
    period_summary.to_csv(TABLES_DIR / "pre_post_llm_period_summary.csv", index=False)
    return period_summary


def build_exclusion_table(calibration_dir: Path, analysis_dir: Path) -> pd.DataFrame:
    config = json.loads((calibration_dir / "frozen_config.json").read_text(encoding="utf-8"))
    repo_summaries = []
    for summary_file in sorted((Path("data/oss")).glob("*/summary.json")):
        repo_summaries.append(json.loads(summary_file.read_text(encoding="utf-8")))
    repo_df = pd.DataFrame(repo_summaries)

    exclusions = pd.DataFrame(
        [
            {"type": "excluded_file_categories", "value": ",".join(config["excluded_file_categories"])},
            {"type": "excluded_length_buckets", "value": ",".join(config["excluded_length_buckets"])},
            {"type": "threshold", "value": config["threshold"]},
            {"type": "allowed_combinations", "value": len(config["allowed_combinations"])},
        ]
    )
    exclusions.to_csv(TABLES_DIR / "exclusions_and_thresholds.csv", index=False)
    repo_df.to_csv(TABLES_DIR / "repo_sample_sizes.csv", index=False)
    return repo_df


def write_summary_markdown(
    calibration_dir: Path,
    overall_metrics: dict,
    pooled_df: pd.DataFrame,
    repo_df: pd.DataFrame,
    pre_post_df: pd.DataFrame,
) -> None:
    start_fraction = pooled_df.iloc[0]["commit_ai_fraction"] if not pooled_df.empty else 0.0
    end_fraction = pooled_df.iloc[-1]["commit_ai_fraction"] if not pooled_df.empty else 0.0
    pre_period = pre_post_df.loc[
        pre_post_df["period"] == "2021-2022 pre-mainstream LLM",
        "weighted_commit_ai_fraction",
    ]
    post_period = pre_post_df.loc[
        pre_post_df["period"] == "2023-2025 post-mainstream LLM",
        "weighted_commit_ai_fraction",
    ]
    pre_post_line = ""
    if not pre_period.empty and not post_period.empty:
        pre_post_line = (
            f"- Better-supported pre/post comparison: `2021-2022 = {pre_period.iloc[0]:.3f}` "
            f"vs `2023-2025 = {post_period.iloc[0]:.3f}` at the medium threshold.\n"
            "- Pre-2020 observations are too sparse for a strong standalone baseline claim.\n"
        )
    summary = f"""# Draft Results Summary

## RQ1: Detector Validity
- Balanced accuracy: {overall_metrics['balanced_accuracy']:.3f}
- ROC AUC: {overall_metrics['roc_auc']:.3f}
- Precision: {overall_metrics['precision']:.3f}
- Recall: {overall_metrics['recall']:.3f}
- Selected threshold: {overall_metrics['threshold']:.2f}

## RQ2: OSS Adoption Trends
- Earliest pooled monthly AI-like commit fraction: {start_fraction:.3f}
- Latest pooled monthly AI-like commit fraction: {end_fraction:.3f}
- Repositories analyzed: {', '.join(sorted(repo_df['repo'].tolist())) if not repo_df.empty else 'n/a'}
{pre_post_line}

## RQ3: Robustness and Commit Context
- Threshold sensitivity and per-repository heterogeneity are captured in the robustness tables and plots.
- Main claims should be restricted to allowed language/file-category/length strata from the frozen calibration config.

## Interpretation
- Results indicate AI-like code signals rather than definitive AI authorship.
- Calibration should be foregrounded before any repository-scale trend claim.
"""
    (ASSETS_DIR / "draft_results_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    calibration_dir = Path("data/calibration_results")
    analysis_dir = Path("data/analysis")

    overall_metrics = json.loads((calibration_dir / "overall_metrics.json").read_text(encoding="utf-8"))
    plot_calibration_by_language(calibration_dir)
    plot_calibration_by_category(calibration_dir)
    pooled_df = plot_pooled_trend(analysis_dir)
    plot_repo_trends(analysis_dir)
    plot_threshold_robustness(analysis_dir)
    pre_post_df = plot_pre_post_llm_trend(analysis_dir)
    repo_df = build_exclusion_table(calibration_dir, analysis_dir)
    write_summary_markdown(calibration_dir, overall_metrics, pooled_df, repo_df, pre_post_df)


if __name__ == "__main__":
    main()
