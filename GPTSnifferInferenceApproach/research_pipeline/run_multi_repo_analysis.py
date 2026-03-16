#!/usr/bin/env python3
"""Run calibrated detector inference on OSS hunk datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from research_pipeline.gptsniffer_detector import CodeBERTLogRegDetector  # noqa: E402


DEFAULT_OSS_DIR = Path("data/oss")
DEFAULT_CALIBRATION_DIR = Path("data/calibration_results")
DEFAULT_OUTPUT_DIR = Path("data/analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-repo AI-likeness analysis.")
    parser.add_argument("--oss-dir", default=str(DEFAULT_OSS_DIR))
    parser.add_argument("--calibration-dir", default=str(DEFAULT_CALIBRATION_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-commits-per-month", type=int, default=60)
    parser.add_argument(
        "--repos",
        nargs="+",
        default=["airflow", "django", "elasticsearch", "spring-boot"],
    )
    return parser.parse_args()


def is_allowed_row(row: pd.Series, allowed_combinations: list[dict]) -> bool:
    for combo in allowed_combinations:
        if (
            row["language"] == combo["language"]
            and row["file_category"] == combo["file_category"]
            and row["length_bucket"] == combo["length_bucket"]
        ):
            return True
    return False


def passes_main_filters(row: pd.Series, frozen_config: dict) -> bool:
    return (
        row["language"] in frozen_config["main_languages"]
        and row["file_category"] in frozen_config["main_file_categories"]
        and row["length_bucket"] not in frozen_config["excluded_length_buckets"]
    )


def analyze_repo(
    repo_name: str,
    detector: CodeBERTLogRegDetector,
    frozen_config: dict,
    oss_dir: Path,
    output_dir: Path,
    max_commits_per_month: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    repo_dir = oss_dir / repo_name
    commit_df = pd.read_parquet(repo_dir / "commit_metrics.parquet")
    hunk_df = pd.read_parquet(repo_dir / "hunk_metrics.parquet")
    commit_df["calendar_month"] = (
        pd.to_datetime(commit_df["author_date"], utc=True).dt.to_period("M").astype(str)
    )
    sampled_commits = (
        commit_df.groupby("calendar_month", group_keys=False)[commit_df.columns.tolist()]
        .apply(
            lambda group: group.sample(
                n=min(len(group), max_commits_per_month),
                random_state=42,
            )
        )
        .reset_index(drop=True)
    )
    sampled_hashes = set(sampled_commits["commit_hash"].tolist())
    hunk_df = hunk_df[hunk_df["commit_hash"].isin(sampled_hashes)].copy()

    hunk_df["length_bucket"] = pd.cut(
        hunk_df["line_count"],
        bins=[-1, 4, 14, 39, 10_000],
        labels=["very_short", "short", "medium", "long"],
    ).astype(str)

    hunk_df = hunk_df[
        hunk_df.apply(lambda row: passes_main_filters(row, frozen_config), axis=1)
    ].copy()

    if hunk_df.empty:
        empty_monthly = pd.DataFrame(
            columns=[
                "calendar_month",
                "commits",
                "commits_with_allowed_hunks",
                "ai_like_commits",
                "mean_ai_line_fraction",
                "mean_max_hunk_score",
                "repo",
                "threshold_label",
                "commit_ai_fraction",
            ]
        )
        return sampled_commits, empty_monthly

    hunk_df["prob_ai"] = detector.predict_proba(hunk_df["snippet"].tolist())
    thresholds = {
        "low": max(0.0, frozen_config["threshold"] - 0.10),
        "medium": frozen_config["threshold"],
        "high": min(0.99, frozen_config["threshold"] + 0.10),
    }

    commit_agg = (
        hunk_df.groupby("commit_hash")
        .agg(
            allowed_hunks=("prob_ai", "size"),
            allowed_hunk_lines=("line_count", "sum"),
            max_hunk_score=("prob_ai", "max"),
            mean_hunk_score=("prob_ai", "mean"),
        )
        .reset_index()
    )

    commit_analysis = sampled_commits.merge(commit_agg, on="commit_hash", how="left")
    commit_analysis["allowed_hunks"] = commit_analysis["allowed_hunks"].fillna(0).astype(int)
    commit_analysis["allowed_hunk_lines"] = commit_analysis["allowed_hunk_lines"].fillna(0).astype(int)
    commit_analysis["max_hunk_score"] = commit_analysis["max_hunk_score"].fillna(0.0)
    commit_analysis["mean_hunk_score"] = commit_analysis["mean_hunk_score"].fillna(0.0)
    for label, threshold in thresholds.items():
        ai_hits = (
            hunk_df[hunk_df["prob_ai"] >= threshold]
            .groupby("commit_hash")
            .agg(
                ai_hunk_count=("prob_ai", "size"),
                ai_hunk_lines=("line_count", "sum"),
            )
            .reset_index()
        )
        ai_hits = ai_hits.rename(
            columns={
                "ai_hunk_count": f"ai_hunk_count_{label}",
                "ai_hunk_lines": f"ai_hunk_lines_{label}",
            }
        )
        commit_analysis = commit_analysis.merge(ai_hits, on="commit_hash", how="left")
        commit_analysis[f"ai_hunk_count_{label}"] = (
            commit_analysis[f"ai_hunk_count_{label}"].fillna(0).astype(int)
        )
        commit_analysis[f"ai_hunk_lines_{label}"] = (
            commit_analysis[f"ai_hunk_lines_{label}"].fillna(0).astype(int)
        )
        commit_analysis[f"has_ai_like_commit_{label}"] = commit_analysis[f"ai_hunk_count_{label}"] > 0
        commit_analysis[f"ai_line_fraction_{label}"] = commit_analysis.apply(
            lambda row: row[f"ai_hunk_lines_{label}"] / row["allowed_hunk_lines"]
            if row["allowed_hunk_lines"] > 0
            else 0.0,
            axis=1,
        )

    repo_output = output_dir / repo_name
    repo_output.mkdir(parents=True, exist_ok=True)
    hunk_df.to_parquet(repo_output / "hunk_predictions.parquet", index=False)
    commit_analysis.to_parquet(repo_output / "commit_analysis.parquet", index=False)

    monthly_rows = []
    for threshold_label in thresholds:
        summary = (
            commit_analysis.groupby("calendar_month")
            .agg(
                commits=("commit_hash", "size"),
                commits_with_allowed_hunks=("allowed_hunks", lambda s: int((s > 0).sum())),
                ai_like_commits=(f"has_ai_like_commit_{threshold_label}", "sum"),
                mean_ai_line_fraction=(f"ai_line_fraction_{threshold_label}", "mean"),
                mean_max_hunk_score=("max_hunk_score", "mean"),
            )
            .reset_index()
        )
        summary["repo"] = repo_name
        summary["threshold_label"] = threshold_label
        summary["commit_ai_fraction"] = summary["ai_like_commits"] / summary["commits"]
        monthly_rows.append(summary)

    monthly_df = pd.concat(monthly_rows, ignore_index=True)
    monthly_df.to_csv(repo_output / "monthly_summary.csv", index=False)
    return commit_analysis, monthly_df


def main() -> None:
    args = parse_args()
    oss_dir = Path(args.oss_dir)
    calibration_dir = Path(args.calibration_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = CodeBERTLogRegDetector.load(calibration_dir / "detector")
    frozen_config = json.loads((calibration_dir / "frozen_config.json").read_text(encoding="utf-8"))

    all_commits = []
    all_monthly = []
    for repo_name in args.repos:
        commit_analysis, monthly_df = analyze_repo(
            repo_name=repo_name,
            detector=detector,
            frozen_config=frozen_config,
            oss_dir=oss_dir,
            output_dir=output_dir,
            max_commits_per_month=args.max_commits_per_month,
        )
        commit_analysis["repo"] = repo_name
        all_commits.append(commit_analysis)
        all_monthly.append(monthly_df)

    combined_commits = pd.concat(all_commits, ignore_index=True)
    combined_monthly = pd.concat(all_monthly, ignore_index=True)
    combined_commits.to_parquet(output_dir / "all_commit_analysis.parquet", index=False)
    combined_monthly.to_csv(output_dir / "all_monthly_summary.csv", index=False)

    pooled_rows = []
    for threshold_label in sorted(combined_monthly["threshold_label"].unique()):
        pooled = (
            combined_commits.groupby("calendar_month")
            .agg(
                commits=("commit_hash", "size"),
                ai_like_commits=(f"has_ai_like_commit_{threshold_label}", "sum"),
                mean_ai_line_fraction=(f"ai_line_fraction_{threshold_label}", "mean"),
                mean_max_hunk_score=("max_hunk_score", "mean"),
            )
            .reset_index()
        )
        pooled["threshold_label"] = threshold_label
        pooled["commit_ai_fraction"] = pooled["ai_like_commits"] / pooled["commits"]
        pooled_rows.append(pooled)
    pooled_df = pd.concat(pooled_rows, ignore_index=True)
    pooled_df.to_csv(output_dir / "pooled_monthly_summary.csv", index=False)


if __name__ == "__main__":
    main()
