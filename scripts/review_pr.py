#!/usr/bin/env python3
"""Post a Claude review comment on a GitHub pull request."""
from __future__ import annotations

import os
import sys
from typing import Iterable

from anthropic import Anthropic
from github import Github


MAX_PATCH_CHARS = 12000
MAX_FILES = 50
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _summarize_files(files: Iterable) -> str:
    chunks: list[str] = []
    total_chars = 0
    for index, file in enumerate(files):
        if index >= MAX_FILES:
            chunks.append("\n[Diff truncated: too many files]")
            break
        header = f"\nFile: {file.filename} (status: {file.status})\n"
        patch = file.patch or "[No patch available]"
        remaining = MAX_PATCH_CHARS - total_chars - len(header)
        if remaining <= 0:
            chunks.append("\n[Diff truncated: size limit reached]")
            break
        if len(patch) > remaining:
            patch = patch[:remaining] + "\n[Diff truncated]"
        chunks.append(header + patch)
        total_chars += len(header) + len(patch)
        if total_chars >= MAX_PATCH_CHARS:
            chunks.append("\n[Diff truncated: size limit reached]")
            break
    return "\n".join(chunks).strip()


def main() -> int:
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: review_pr.py <pr_number>")

    pr_number = int(sys.argv[1])
    repo_name = _require_env("GITHUB_REPOSITORY")
    github_token = _require_env("GITHUB_TOKEN")
    anthropic_key = _require_env("ANTHROPIC_API_KEY")
    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)

    github = Github(github_token)
    repo = github.get_repo(repo_name)
    pr = repo.get_pull(pr_number)

    files = list(pr.get_files())
    diff_summary = _summarize_files(files)

    prompt = (
        "You are an expert code reviewer. Provide a concise review for the PR below. "
        "Focus on correctness, risks, and actionable improvements. "
        "Return Markdown with sections: Summary, Issues, Suggestions.\n\n"
        f"Repository: {repo_name}\n"
        f"PR: #{pr.number} {pr.title}\n\n"
        f"Description:\n{pr.body or '[No description provided]'}\n\n"
        f"Diff:\n{diff_summary}\n"
    )

    client = Anthropic(api_key=anthropic_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    review_text = response.content[0].text.strip()
    pr.create_issue_comment(review_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
