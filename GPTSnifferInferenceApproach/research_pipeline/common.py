from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


BOT_MARKERS = (
    "[bot]",
    "dependabot",
    "renovate",
    "github-actions",
    "claude",
    "codex",
    "cursor",
    "devin",
    "copilot",
)

DOC_EXTENSIONS = {
    ".md",
    ".mdx",
    ".rst",
    ".txt",
    ".adoc",
}

CONFIG_EXTENSIONS = {
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".properties",
    ".json",
    ".xml",
}

SOURCE_EXTENSIONS = {
    ".py": "Python",
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".go": "Go",
    ".rb": "Ruby",
    ".rs": "Rust",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".scala": "Scala",
    ".c": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".cxx": "C++",
    ".h": "C/C++ Header",
    ".hpp": "C/C++ Header",
}

LOCKFILE_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "cargo.lock",
    "pdm.lock",
    "composer.lock",
    "gradle.lockfile",
}

GENERATED_PATTERNS = (
    "/vendor/",
    "/vendors/",
    "/node_modules/",
    "/dist/",
    "/build/",
    "/generated/",
    "/__generated__/",
    ".min.js",
    ".min.css",
)

TEST_MARKERS = (
    "/test/",
    "/tests/",
    "/testing/",
    "/__tests__/",
    "test_",
    "_test.",
    "tests.py",
    "spec/",
)

DOC_MARKERS = ("/docs/", "/doc/", "readme", "changelog", "license", "notice")
CONFIG_MARKERS = (".github/", ".gitlab/", "docker-compose", "pyproject.toml", "setup.cfg")

CODE_LIKE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[{}();=\[\].]")


def normalize_repo_url(url: str | None) -> str | None:
    if not url:
        return None
    normalized = url.strip()
    normalized = normalized.replace("https://api.github.com/repos/", "https://github.com/")
    normalized = normalized.removesuffix(".git")
    return normalized


def repo_full_name_from_url(url: str | None) -> str | None:
    normalized = normalize_repo_url(url)
    if not normalized or "github.com/" not in normalized:
        return None
    return normalized.split("github.com/", 1)[1]


def is_bot_identity(*values: str | None) -> bool:
    haystack = " ".join(v.lower() for v in values if v)
    return any(marker in haystack for marker in BOT_MARKERS)


def detect_language(filename: str, repo_language: str | None = None) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in SOURCE_EXTENSIONS:
        return SOURCE_EXTENSIONS[suffix]
    return repo_language or "Unknown"


def classify_file_category(filename: str) -> str:
    normalized = filename.lower().replace("\\", "/")
    name = Path(normalized).name
    suffix = Path(normalized).suffix.lower()

    if name in LOCKFILE_NAMES:
        return "lockfile"
    if any(marker in normalized for marker in DOC_MARKERS) or suffix in DOC_EXTENSIONS:
        return "docs"
    if any(marker in normalized for marker in TEST_MARKERS):
        return "test"
    if any(marker in normalized for marker in CONFIG_MARKERS) or suffix in CONFIG_EXTENSIONS:
        return "config"
    return "source"


def is_generated_or_vendored(filename: str) -> bool:
    normalized = filename.lower().replace("\\", "/")
    return any(pattern in normalized for pattern in GENERATED_PATTERNS)


def is_excluded_file(filename: str) -> bool:
    category = classify_file_category(filename)
    return category == "lockfile" or is_generated_or_vendored(filename)


def snippet_line_count(snippet: str) -> int:
    return sum(1 for line in snippet.splitlines() if line.strip())


def length_bucket(line_count: int) -> str:
    if line_count < 5:
        return "very_short"
    if line_count < 15:
        return "short"
    if line_count < 40:
        return "medium"
    return "long"


def has_code_like_signal(lines: Iterable[str]) -> bool:
    return any(CODE_LIKE_PATTERN.search(line) for line in lines if line.strip())


def clean_added_line(line: str) -> str:
    if line.startswith("+") and not line.startswith("+++"):
        return line[1:]
    return line


def extract_added_hunks_from_patch(patch: str | None) -> list[dict]:
    if not patch:
        return []

    hunks: list[dict] = []
    current: list[str] = []

    def flush_current() -> None:
        nonlocal current
        if not current:
            return
        if has_code_like_signal(current):
            snippet = "\n".join(current).strip("\n")
            if snippet.strip():
                hunks.append(
                    {
                        "snippet": snippet,
                        "line_count": snippet_line_count(snippet),
                        "char_count": len(snippet),
                    }
                )
        current = []

    for raw_line in patch.splitlines():
        if raw_line.startswith("@@"):
            flush_current()
            continue
        if raw_line.startswith("+++"):
            continue
        if raw_line.startswith("+"):
            current.append(clean_added_line(raw_line))
            continue
        flush_current()

    flush_current()
    return hunks


def split_unified_diff_by_file(diff_text: str | None) -> list[dict]:
    if not diff_text:
        return []

    files: list[dict] = []
    current_lines: list[str] = []
    current_filename: str | None = None

    def flush_current() -> None:
        nonlocal current_lines, current_filename
        if current_filename and current_lines:
            files.append({"filename": current_filename, "patch": "\n".join(current_lines)})
        current_lines = []
        current_filename = None

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            flush_current()
            current_lines = [line]
            parts = line.split()
            if len(parts) >= 4 and parts[3].startswith("b/"):
                current_filename = parts[3][2:]
            continue

        if current_filename is None:
            continue

        if line.startswith("+++ b/"):
            current_filename = line[6:]
        current_lines.append(line)

    flush_current()
    return files
