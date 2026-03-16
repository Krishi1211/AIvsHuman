#!/usr/bin/env python3
"""Build a labeled hunk-level calibration dataset from AIDev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from research_pipeline.common import (
    classify_file_category,
    detect_language,
    extract_added_hunks_from_patch,
    is_bot_identity,
    is_excluded_file,
    length_bucket,
    normalize_repo_url,
    repo_full_name_from_url,
    split_unified_diff_by_file,
)


DEFAULT_OUTPUT_DIR = Path("data/calibration")
TARGET_LANGUAGES = {"Python", "Java"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build calibration hunks from AIDev.")
    parser.add_argument("--aidev-dir", default="data/aidev")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-per-stratum", type=int, default=400)
    parser.add_argument("--human-prs-per-language", type=int, default=40)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_ai_pull_requests(aidev_dir: Path) -> pd.DataFrame:
    ai_pr = pd.read_parquet(aidev_dir / "all_pull_request.parquet")[
        ["id", "agent", "created_at", "repo_url", "html_url", "title"]
    ].copy()
    ai_pr["label"] = 1
    ai_pr["repo_url"] = ai_pr["repo_url"].map(normalize_repo_url)
    ai_pr["repo_full_name"] = ai_pr["repo_url"].map(repo_full_name_from_url)
    return ai_pr


def load_human_pull_requests(aidev_dir: Path) -> pd.DataFrame:
    human_pr = pd.read_parquet(aidev_dir / "human_pull_request.parquet")[
        ["id", "agent", "created_at", "repo_url", "html_url", "title", "number"]
    ].copy()
    human_pr["label"] = 0
    human_pr["repo_url"] = human_pr["repo_url"].map(normalize_repo_url)
    human_pr["repo_full_name"] = human_pr["repo_url"].map(repo_full_name_from_url)
    return human_pr


def load_repository_languages(aidev_dir: Path) -> pd.DataFrame:
    repo_df = pd.read_parquet(aidev_dir / "all_repository.parquet")[
        ["url", "full_name", "language", "stars"]
    ].copy()
    repo_df["repo_url"] = repo_df["url"].map(normalize_repo_url)
    return repo_df[["repo_url", "language", "stars"]]


def build_hunk_rows(merged_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for record in merged_df.itertuples(index=False):
        filename = getattr(record, "filename")
        if not isinstance(filename, str) or is_excluded_file(filename):
            continue

        if is_bot_identity(getattr(record, "author"), getattr(record, "committer")):
            continue

        file_category = classify_file_category(filename)
        language = detect_language(filename, getattr(record, "language"))
        if language not in TARGET_LANGUAGES:
            continue

        hunks = extract_added_hunks_from_patch(getattr(record, "patch"))
        for hunk_index, hunk in enumerate(hunks):
            rows.append(
                {
                    "label": int(getattr(record, "label")),
                    "agent": getattr(record, "agent"),
                    "pr_id": int(getattr(record, "pr_id")),
                    "sha": getattr(record, "sha"),
                    "repo_url": getattr(record, "repo_url"),
                    "repo_full_name": getattr(record, "repo_full_name"),
                    "repo_language": getattr(record, "language"),
                    "stars": getattr(record, "stars"),
                    "created_at": getattr(record, "created_at"),
                    "html_url": getattr(record, "html_url"),
                    "title": getattr(record, "title"),
                    "filename": filename,
                    "status": getattr(record, "status"),
                    "file_category": file_category,
                    "language": language,
                    "commit_author": getattr(record, "author"),
                    "commit_committer": getattr(record, "committer"),
                    "commit_message": getattr(record, "message"),
                    "hunk_index": hunk_index,
                    "snippet": hunk["snippet"],
                    "line_count": hunk["line_count"],
                    "char_count": hunk["char_count"],
                    "length_bucket": length_bucket(hunk["line_count"]),
                }
            )
    return rows


def fetch_url_text(url: str, pause_s: float = 0.1) -> str | None:
    request = urllib.request.Request(url, headers={"User-Agent": "gitdiffstory-research"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8", errors="replace")
        time.sleep(pause_s)
        return text
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def build_human_hunk_rows(
    human_pr_df: pd.DataFrame,
    repo_df: pd.DataFrame,
    human_prs_per_language: int,
    random_state: int,
) -> list[dict]:
    human_df = human_pr_df.merge(repo_df, on="repo_url", how="left")
    human_df = human_df[human_df["language"].isin(TARGET_LANGUAGES)].copy()
    human_df["stars"] = human_df["stars"].fillna(0)

    selected_frames = []
    for language in sorted(TARGET_LANGUAGES):
        language_df = human_df[human_df["language"] == language]
        subset = language_df.sample(
            n=min(len(language_df), human_prs_per_language),
            random_state=random_state,
        )
        selected_frames.append(subset)
    selected_human_df = pd.concat(selected_frames, ignore_index=True)

    rows: list[dict] = []
    for index, record in enumerate(selected_human_df.itertuples(index=False), start=1):
        if index % 10 == 0:
            print(f"Fetched {index}/{len(selected_human_df)} human PR patches...")
        patch_url = f"{getattr(record, 'html_url')}.patch"
        patch_text = fetch_url_text(patch_url)
        if not patch_text:
            continue

        file_patches = split_unified_diff_by_file(patch_text)
        for file_patch in file_patches:
            filename = file_patch["filename"]
            if not filename or is_excluded_file(filename):
                continue

            file_category = classify_file_category(filename)
            language = detect_language(filename, getattr(record, "language"))
            if language not in TARGET_LANGUAGES:
                continue

            hunks = extract_added_hunks_from_patch(file_patch["patch"])
            for hunk_index, hunk in enumerate(hunks):
                rows.append(
                    {
                        "label": 0,
                        "agent": getattr(record, "agent"),
                        "pr_id": int(getattr(record, "id")),
                        "sha": None,
                        "repo_url": getattr(record, "repo_url"),
                        "repo_full_name": getattr(record, "repo_full_name"),
                        "repo_language": getattr(record, "language"),
                        "stars": getattr(record, "stars"),
                        "created_at": getattr(record, "created_at"),
                        "html_url": getattr(record, "html_url"),
                        "title": getattr(record, "title"),
                        "filename": filename,
                        "status": "patch",
                        "file_category": file_category,
                        "language": language,
                        "commit_author": None,
                        "commit_committer": None,
                        "commit_message": None,
                        "hunk_index": hunk_index,
                        "snippet": hunk["snippet"],
                        "line_count": hunk["line_count"],
                        "char_count": hunk["char_count"],
                        "length_bucket": length_bucket(hunk["line_count"]),
                    }
                )
    return rows


def sample_by_strata(df: pd.DataFrame, max_per_stratum: int, random_state: int) -> pd.DataFrame:
    strata_cols = ["label", "language", "file_category", "length_bucket"]
    sampled = (
        df.groupby(strata_cols, group_keys=False)[df.columns.tolist()]
        .apply(
            lambda group: group.sample(
                n=min(len(group), max_per_stratum), random_state=random_state
            )
        )
        .reset_index(drop=True)
    )
    return sampled


def write_summary(df: pd.DataFrame, sampled_df: pd.DataFrame, output_dir: Path) -> None:
    full_counts = (
        df.groupby(["label", "language", "file_category", "length_bucket"])
        .size()
        .reset_index(name="count")
        .sort_values(["label", "language", "file_category", "length_bucket"])
    )
    sampled_counts = (
        sampled_df.groupby(["label", "language", "file_category", "length_bucket"])
        .size()
        .reset_index(name="count")
        .sort_values(["label", "language", "file_category", "length_bucket"])
    )

    full_counts.to_csv(output_dir / "calibration_hunks_full_counts.csv", index=False)
    sampled_counts.to_csv(output_dir / "calibration_hunks_sampled_counts.csv", index=False)

    summary = {
        "full_rows": int(len(df)),
        "sampled_rows": int(len(sampled_df)),
        "labels": {
            str(label): int(count) for label, count in df["label"].value_counts().sort_index().items()
        },
        "sampled_labels": {
            str(label): int(count)
            for label, count in sampled_df["label"].value_counts().sort_index().items()
        },
        "languages": sorted(df["language"].dropna().unique().tolist()),
        "file_categories": sorted(df["file_category"].dropna().unique().tolist()),
    }
    (output_dir / "calibration_hunks_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    aidev_dir = Path(args.aidev_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ai_pr_df = load_ai_pull_requests(aidev_dir)
    human_pr_df = load_human_pull_requests(aidev_dir)
    repo_df = load_repository_languages(aidev_dir)
    commit_df = pd.read_parquet(aidev_dir / "pr_commit_details.parquet")

    merged_df = commit_df.merge(ai_pr_df, left_on="pr_id", right_on="id", how="inner")
    merged_df = merged_df.merge(repo_df, on="repo_url", how="left")

    ai_rows = build_hunk_rows(merged_df)
    human_rows = build_human_hunk_rows(
        human_pr_df=human_pr_df,
        repo_df=repo_df,
        human_prs_per_language=args.human_prs_per_language,
        random_state=args.random_state,
    )
    calibration_df = pd.DataFrame(ai_rows + human_rows)
    calibration_df = calibration_df[calibration_df["line_count"] > 0].reset_index(drop=True)

    sampled_df = sample_by_strata(
        calibration_df,
        max_per_stratum=args.max_per_stratum,
        random_state=args.random_state,
    )

    calibration_df.to_parquet(output_dir / "calibration_hunks_full.parquet", index=False)
    sampled_df.to_parquet(output_dir / "calibration_hunks_sampled.parquet", index=False)
    write_summary(calibration_df, sampled_df, output_dir)

    print(f"Full calibration rows: {len(calibration_df)}")
    print(f"Sampled calibration rows: {len(sampled_df)}")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
