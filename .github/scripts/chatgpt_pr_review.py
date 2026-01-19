#!/usr/bin/env python3
"""
ChatGPT/Codex PR Review - Automated code review using OpenAI.
Analyzes pull requests and provides detailed feedback.
"""

import os
import subprocess

from github import Github
from openai import OpenAI


def get_pr_diff() -> str:
    """Get the diff for the current PR."""
    try:
        with open("/tmp/pr_diff.txt") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def get_changed_files() -> list[str]:
    """Get list of changed files."""
    try:
        base_ref = os.environ.get("GITHUB_BASE_REF", "main")
        result = subprocess.run(
            ["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [f for f in result.stdout.strip().split("\n") if f]
    except subprocess.CalledProcessError:
        return []


def main():
    # Configure OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping ChatGPT review")
        return

    client = OpenAI(api_key=api_key)

    # Get GitHub client
    gh_token = os.environ.get("GITHUB_TOKEN")
    if not gh_token:
        print("GITHUB_TOKEN not set")
        return

    gh = Github(gh_token)
    repo = gh.get_repo(os.environ.get("GITHUB_REPOSITORY"))

    pr_number = os.environ.get("PR_NUMBER")
    if not pr_number:
        print("PR_NUMBER not set")
        return

    pr = repo.get_pull(int(pr_number))

    # Get diff and context
    diff = get_pr_diff()
    changed_files = get_changed_files()
    pr_title = os.environ.get("PR_TITLE", pr.title)
    pr_body = os.environ.get("PR_BODY", pr.body) or "No description"

    if not diff:
        print("No diff found")
        return

    # Truncate diff if too long
    max_diff_length = 25000
    if len(diff) > max_diff_length:
        diff = diff[:max_diff_length] + "\n... (diff truncated)"

    # Build review prompt
    system_prompt = """You are an expert code reviewer powered by OpenAI.
Your task is to review pull requests and provide constructive, actionable feedback.
Focus on code quality, potential bugs, security issues, and best practices.
Format your response in GitHub-flavored markdown."""

    user_prompt = f"""Review this pull request:

## Pull Request
**Title:** {pr_title}
**Description:** {pr_body}

## Changed Files
{chr(10).join(f'- {f}' for f in changed_files)}

## Diff
```diff
{diff}
```

Provide a thorough review covering:
1. **Summary**: What does this PR do?
2. **Code Quality**: Readability, maintainability, best practices
3. **Potential Issues**: Bugs, edge cases, security concerns
4. **Suggestions**: Specific improvements
5. **Overall**: Approve, request changes, or comment"""

    try:
        # Generate review using GPT-4
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2000,
            temperature=0.3,
        )

        review_text = response.choices[0].message.content

        # Post review comment
        review_body = f"""## ChatGPT Codex Review

{review_text}

---
*Automated review by OpenAI ChatGPT/Codex*
"""

        pr.create_issue_comment(review_body)
        print("ChatGPT review posted successfully")

    except Exception as e:
        print(f"Error generating review: {e}")
        pr.create_issue_comment(
            f"ChatGPT Code Review encountered an error: {str(e)}"
        )


if __name__ == "__main__":
    main()
