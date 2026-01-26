#!/usr/bin/env python3
"""
Claude PR Review - Automated code review using Anthropic Claude.
Analyzes pull requests and provides detailed feedback.
"""

import os
import subprocess
import sys

import anthropic
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
    except subprocess.CalledProcessError as e:
        print(f"Error getting diff: {e}")
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
    except subprocess.CalledProcessError as e:
        print(f"Error getting changed files: {e}")
        return []


def main():
    # Get PR number from command line or environment
    pr_number = None
    if len(sys.argv) > 1:
        pr_number = sys.argv[1]
    else:
        pr_number = os.environ.get("PR_NUMBER")

    if not pr_number:
        print("PR number not provided. Usage: python claude_pr_review.py <pr_number>")
        return

    # Configure Anthropic client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set, skipping Claude review")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Get GitHub client
    gh_token = os.environ.get("GITHUB_TOKEN")
    if not gh_token:
        print("GITHUB_TOKEN not set")
        return

    gh = Github(gh_token)
    repo_name = os.environ.get("GITHUB_REPOSITORY")
    if not repo_name:
        print("GITHUB_REPOSITORY not set")
        return

    repo = gh.get_repo(repo_name)
    try:
        pr = repo.get_pull(int(pr_number))
    except ValueError:
        print(f"Error: PR number '{pr_number}' is not a valid integer.")
        return

    # Get diff and files
    diff = get_pr_diff()
    changed_files = get_changed_files()

    if not diff:
        print("No diff found")
        return

    # Truncate diff if too long (Claude has generous token limits but we should be reasonable)
    max_diff_length = 50000
    truncated = False
    if len(diff) > max_diff_length:
        diff = diff[:max_diff_length]
        truncated = True

    # Build review prompt
    system_prompt = """You are an expert code reviewer powered by Anthropic Claude.
Your task is to review pull requests and provide constructive, actionable feedback.
Focus on code quality, potential bugs, security issues, and best practices.
Format your response in GitHub-flavored markdown.
Be thorough but concise. Prioritize the most important issues."""

    files_list = "\n".join(f"- {f}" for f in changed_files)
    truncated_note = "(Note: diff was truncated due to size)" if truncated else ""

    user_prompt = f"""Review this pull request:

## Pull Request
**Title:** {pr.title}
**Description:** {pr.body or 'No description provided'}

## Changed Files
{files_list}

## Diff
```diff
{diff}
```
{truncated_note}

Please provide a thorough code review covering:

1. **Summary**: Brief overview of what this PR does
2. **Code Quality**: Evaluate readability, maintainability, and best practices
3. **Potential Issues**: Bugs, edge cases, security concerns
4. **Suggestions**: Specific improvements with code examples if helpful
5. **Overall Assessment**: Approve, request changes, or comment

Be constructive and specific. Use inline code references where applicable."""

    try:
        # Generate review using Claude
        response = client.messages.create(
            model="claude-3.5-sonnet-20240620",
            max_tokens=2000,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            system=system_prompt,
        )

        review_text = response.content[0].text

        # Post review comment
        review_body = f"""## Claude Code Review

{review_text}

---
*Automated review by Anthropic Claude*
"""

        pr.create_issue_comment(review_body)
        print("Claude review posted successfully")

    except (anthropic.APIError, Exception) as e:
        error_msg = f"Claude review failed: {e}"
        print(error_msg)
        pr.create_issue_comment(
            f"Claude Code Review encountered an error: {error_msg}"
        )


if __name__ == "__main__":
    main()
