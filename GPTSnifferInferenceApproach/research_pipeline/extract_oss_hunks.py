#!/usr/bin/env python3
"""Extract filtered commit and hunk data from OSS repositories."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from pydriller import Repository

sys.path.append(str(Path(__file__).resolve().parents[1]))

from research_pipeline.common import (  # noqa: E402
    classify_file_category,
    detect_language,
    extract_added_hunks_from_patch,
    is_bot_identity,
    is_excluded_file,
)


REPOS_DIR = Path("repos")
OUTPUT_DIR = Path("data/oss")


@dataclass(frozen=True)
class RepoConfig:
    slug: str
    url: str
    branch: str = "main"


REPO_CONFIGS = {
    "airflow": RepoConfig("apache_airflow", "https://github.com/apache/airflow.git"),
    "django": RepoConfig("django_django", "https://github.com/django/django.git"),
    "elasticsearch": RepoConfig(
        "elastic_elasticsearch", "https://github.com/elastic/elasticsearch.git"
    ),
    "spring-boot": RepoConfig(
        "spring_projects_spring_boot",
        "https://github.com/spring-projects/spring-boot.git",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract filtered OSS hunks.")
    parser.add_argument(
        "--repos",
        nargs="+",
        default=["airflow", "django", "elasticsearch", "spring-boot"],
    )
    parser.add_argument("--since", default="2021-01-01")
    parser.add_argument("--to", default="2025-12-31T23:59:59")
    return parser.parse_args()


def run_git_command(args: list[str]) -> None:
    subprocess.run(args, check=True)


def ensure_repo(repo: RepoConfig) -> Path:
    repo_path = REPOS_DIR / repo.slug
    REPOS_DIR.mkdir(parents=True, exist_ok=True)

    if (repo_path / ".git").exists():
        print(f"Updating {repo.slug}...")
        run_git_command(["git", "-C", str(repo_path), "fetch", "--all", "--tags", "--prune"])
        return repo_path

    if repo_path.exists():
        shutil.rmtree(repo_path)

    print(f"Cloning {repo.url}...")
    run_git_command(
        [
            "git",
            "-c",
            "http.version=HTTP/1.1",
            "clone",
            "--single-branch",
            "--branch",
            repo.branch,
            "--no-checkout",
            repo.url,
            str(repo_path),
        ]
    )
    return repo_path


def extract_repo(repo_name: str, repo_path: Path, since: datetime, to: datetime) -> None:
    repo_output = OUTPUT_DIR / repo_name
    repo_output.mkdir(parents=True, exist_ok=True)

    commit_rows: list[dict] = []
    hunk_rows: list[dict] = []

    for commit in Repository(str(repo_path), since=since, to=to).traverse_commits():
        parents = len(commit.parents)
        if parents > 1:
            continue

        author_name = getattr(commit.author, "name", None)
        author_email = getattr(commit.author, "email", None)
        committer_name = getattr(commit.committer, "name", None)
        committer_email = getattr(commit.committer, "email", None)

        if is_bot_identity(author_name, author_email, committer_name, committer_email):
            continue

        repo_hunk_count = 0
        repo_added_lines = 0
        source_file_count = 0
        test_file_count = 0
        docs_file_count = 0
        config_file_count = 0

        for modified_file in commit.modified_files:
            filename = modified_file.new_path or modified_file.old_path or modified_file.filename
            if not filename or is_excluded_file(filename):
                continue

            file_category = classify_file_category(filename)
            language = detect_language(filename)
            patch = modified_file.diff

            file_hunks = extract_added_hunks_from_patch(patch)
            if not file_hunks:
                continue

            if file_category == "source":
                source_file_count += 1
            elif file_category == "test":
                test_file_count += 1
            elif file_category == "docs":
                docs_file_count += 1
            elif file_category == "config":
                config_file_count += 1

            for hunk_index, hunk in enumerate(file_hunks):
                repo_hunk_count += 1
                repo_added_lines += hunk["line_count"]
                hunk_rows.append(
                    {
                        "repo": repo_name,
                        "commit_hash": commit.hash,
                        "author_date": commit.author_date.isoformat(),
                        "filename": filename,
                        "file_category": file_category,
                        "language": language,
                        "hunk_index": hunk_index,
                        "line_count": hunk["line_count"],
                        "char_count": hunk["char_count"],
                        "snippet": hunk["snippet"],
                        "commit_insertions": commit.insertions,
                        "commit_deletions": commit.deletions,
                        "commit_files": commit.files,
                    }
                )

        commit_rows.append(
            {
                "repo": repo_name,
                "commit_hash": commit.hash,
                "author_name": author_name,
                "author_email": author_email,
                "committer_name": committer_name,
                "committer_email": committer_email,
                "author_date": commit.author_date.isoformat(),
                "committer_date": commit.committer_date.isoformat(),
                "message": commit.msg.replace("\n", "\\n"),
                "parents": parents,
                "files": commit.files,
                "insertions": commit.insertions,
                "deletions": commit.deletions,
                "lines": commit.lines,
                "hunk_count": repo_hunk_count,
                "added_hunk_lines": repo_added_lines,
                "source_file_count": source_file_count,
                "test_file_count": test_file_count,
                "docs_file_count": docs_file_count,
                "config_file_count": config_file_count,
            }
        )

        if len(commit_rows) % 1000 == 0:
            print(f"  {repo_name}: processed {len(commit_rows)} commits")

    commit_df = pd.DataFrame(commit_rows)
    hunk_df = pd.DataFrame(hunk_rows)

    commit_df.to_parquet(repo_output / "commit_metrics.parquet", index=False)
    hunk_df.to_parquet(repo_output / "hunk_metrics.parquet", index=False)

    summary = {
        "repo": repo_name,
        "commit_rows": int(len(commit_df)),
        "hunk_rows": int(len(hunk_df)),
        "since": since.isoformat(),
        "to": to.isoformat(),
        "output_dir": str(repo_output.resolve()),
    }
    (repo_output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Finished {repo_name}: {len(commit_df)} commits, {len(hunk_df)} hunks")


def main() -> None:
    args = parse_args()
    since = datetime.fromisoformat(args.since)
    to = datetime.fromisoformat(args.to)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for repo_key in args.repos:
        repo = REPO_CONFIGS[repo_key]
        repo_path = ensure_repo(repo)
        extract_repo(repo_key, repo_path, since, to)


if __name__ == "__main__":
    main()
