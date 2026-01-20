#!/usr/bin/env python3
"""
Gemini PR Review - Automated code review using Google AI Studio.
Analyzes pull requests and provides detailed feedback.
"""

import os
import subprocess

import google.generativeai as genai
from github import Github


def get_pr_diff() -> str:
    """Get the diff for the current PR."""
    try:
        base_ref = os.environ.get("GITHUB_BASE_REF", "main")
        result = subprocess.run(
            ["git", "diff", f"origin/{base_ref}...HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
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
        return result.stdout.strip().split("\n")
    except subprocess.CalledProcessError:
        return []


def main():
    # Configure Gemini
    api_key = os.environ.get("GOOGLE_AI_API_KEY")
    if not api_key:
        print("GOOGLE_AI_API_KEY not set, skipping Gemini review")
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

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

    # Get diff and files
    diff = get_pr_diff()
    changed_files = get_changed_files()

    if not diff:
        print("No diff found")
        return

    # Truncate diff if too long (Gemini has token limits)
    max_diff_length = 30000
    if len(diff) > max_diff_length:
        diff = diff[:max_diff_length] + "\n... (diff truncated)"

    # Build review prompt
    prompt = f"""You are an expert code reviewer. Review this pull request and provide constructive feedback.

## Pull Request
**Title:** {pr.title}
**Description:** {pr.body or 'No description provided'}

## Changed Files
{chr(10).join(f'- {f}' for f in changed_files if f)}

## Diff
```diff
{diff}
```

## Review Instructions
Provide a thorough code review covering:

1. **Summary**: Brief overview of what this PR does
2. **Code Quality**: Evaluate readability, maintainability, and best practices
3. **Potential Issues**: Bugs, edge cases, security concerns
4. **Suggestions**: Specific improvements with code examples if helpful
5. **Overall Assessment**: Approve, request changes, or comment

Format your response in GitHub-flavored markdown. Be constructive and specific.
Use inline code references where applicable."""

    try:
        # Generate review
        response = model.generate_content(prompt)
        review_text = response.text

        # Post review comment
        review_body = f"""## Gemini Code Review

{review_text}

---
*Automated review by Google Gemini AI Studio*
"""

        pr.create_issue_comment(review_body)
        print("Gemini review posted successfully")

    except Exception as e:
        print(f"Error generating review: {e}")
        # Post error as comment
        pr.create_issue_comment(
            "Gemini Code Review encountered an error. Please check the workflow logs for details."
        )


if __name__ == "__main__":
    main()
