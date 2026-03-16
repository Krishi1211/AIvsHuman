#!/usr/bin/env python3
"""Calibrate a GPTSniffer-style detector on labeled AIDev hunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))

from research_pipeline.gptsniffer_detector import CodeBERTLogRegDetector  # noqa: E402


DEFAULT_INPUT = Path("data/calibration/calibration_hunks_sampled.parquet")
DEFAULT_OUTPUT = Path("data/calibration_results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate a CodeBERT detector.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-total-rows", type=int, default=6000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def metrics_for_threshold(labels: pd.Series, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn = int(((preds == 0) & (labels == 0)).sum())
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_strata(
    df: pd.DataFrame,
    group_cols: list[str],
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for key, group in df.groupby(group_cols):
        if len(group) < 20 or group["label"].nunique() < 2:
            continue
        probs = group["prob_ai"].to_numpy()
        labels = group["label"]
        metrics = metrics_for_threshold(labels, probs, threshold)
        row = {"support": int(len(group))}
        if isinstance(key, tuple):
            for col, value in zip(group_cols, key):
                row[col] = value
        else:
            row[group_cols[0]] = key
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def derive_frozen_config(
    predictions_df: pd.DataFrame, threshold: float, combined_df: pd.DataFrame
) -> dict:
    allowed_combinations = []
    for row in combined_df.itertuples(index=False):
        if (
            row.support >= 25
            and row.balanced_accuracy >= 0.60
            and row.false_positive_rate <= 0.35
        ):
            allowed_combinations.append(
                {
                    "language": row.language,
                    "file_category": row.file_category,
                    "length_bucket": row.length_bucket,
                }
            )

    if not allowed_combinations:
        allowed_combinations = [
            {
                "language": language,
                "file_category": file_category,
                "length_bucket": length_bucket,
            }
            for language in sorted(predictions_df["language"].dropna().unique())
            for file_category in sorted(predictions_df["file_category"].dropna().unique())
            for length_bucket in sorted(predictions_df["length_bucket"].dropna().unique())
            if file_category in {"source", "test"} and length_bucket != "very_short"
        ]

    return {
        "threshold": threshold,
        "allowed_combinations": allowed_combinations,
        "main_file_categories": ["source", "test"],
        "main_languages": sorted(predictions_df["language"].dropna().unique().tolist()),
        "excluded_file_categories": ["docs", "config", "lockfile"],
        "excluded_length_buckets": ["very_short"],
        "notes": "Detector outputs are probabilistic AI-likeness scores calibrated on AIDev sampled hunks.",
    }


def capped_stratified_sample(df: pd.DataFrame, max_total_rows: int, random_state: int) -> pd.DataFrame:
    if len(df) <= max_total_rows:
        return df.copy()

    strata_cols = ["label", "language", "file_category", "length_bucket"]
    num_groups = max(df.groupby(strata_cols).ngroups, 1)
    cap_per_group = max(max_total_rows // num_groups, 20)
    sampled = (
        df.groupby(strata_cols, group_keys=False)[df.columns.tolist()]
        .apply(
            lambda group: group.sample(
                n=min(len(group), cap_per_group),
                random_state=random_state,
            )
        )
        .reset_index(drop=True)
    )
    if len(sampled) > max_total_rows:
        sampled = sampled.sample(n=max_total_rows, random_state=random_state).reset_index(drop=True)
    return sampled


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    df = df[df["snippet"].str.len() > 0].reset_index(drop=True)
    df = capped_stratified_sample(df, args.max_total_rows, args.random_state)

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"],
    )

    detector = CodeBERTLogRegDetector()
    detector.fit(train_df["snippet"].tolist(), train_df["label"].tolist())
    detector.save(output_dir / "detector")

    test_probs = detector.predict_proba(test_df["snippet"].tolist())
    test_predictions = test_df.copy()
    test_predictions["prob_ai"] = test_probs

    thresholds = np.arange(0.30, 0.81, 0.05)
    threshold_df = pd.DataFrame(
        [metrics_for_threshold(test_predictions["label"], test_probs, threshold) for threshold in thresholds]
    ).sort_values(["balanced_accuracy", "precision", "recall"], ascending=False)
    best_threshold = float(threshold_df.iloc[0]["threshold"])

    overall = metrics_for_threshold(test_predictions["label"], test_probs, best_threshold)
    overall["roc_auc"] = roc_auc_score(test_predictions["label"], test_probs)
    overall["test_rows"] = int(len(test_predictions))
    overall["train_rows"] = int(len(train_df))

    by_language = evaluate_strata(test_predictions, ["language"], best_threshold)
    by_category = evaluate_strata(test_predictions, ["file_category"], best_threshold)
    by_length = evaluate_strata(test_predictions, ["length_bucket"], best_threshold)
    by_combination = evaluate_strata(
        test_predictions,
        ["language", "file_category", "length_bucket"],
        best_threshold,
    )

    frozen_config = derive_frozen_config(test_predictions, best_threshold, by_combination)

    test_predictions.to_parquet(output_dir / "calibration_predictions.parquet", index=False)
    threshold_df.to_csv(output_dir / "threshold_metrics.csv", index=False)
    by_language.to_csv(output_dir / "metrics_by_language.csv", index=False)
    by_category.to_csv(output_dir / "metrics_by_file_category.csv", index=False)
    by_length.to_csv(output_dir / "metrics_by_length_bucket.csv", index=False)
    by_combination.to_csv(output_dir / "metrics_by_combination.csv", index=False)
    (output_dir / "frozen_config.json").write_text(
        json.dumps(frozen_config, indent=2), encoding="utf-8"
    )
    (output_dir / "overall_metrics.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_predictions)}")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Balanced accuracy: {overall['balanced_accuracy']:.3f}")
    print(f"ROC AUC: {overall['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
