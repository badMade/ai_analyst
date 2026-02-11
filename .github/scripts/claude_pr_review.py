#!/usr/bin/env python3
"""
Claude PR Review - Automated code review using Anthropic Claude.
Analyzes pull requests and provides detailed feedback.
Sets a commit status to block or approve auto-merge.
"""

import os
import re
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


def parse_review_verdict(review_text: str) -> str:
    """Parse the AI review text to determine approve/reject/comment.

    Returns one of: 'approve', 'request_changes', 'comment'
    """
    text_lower = review_text.lower()

    # Look for the Overall Assessment section first
    assessment_match = re.search(
        r"(?:overall\s*assessment|overall)[:\s]*\n?(.*?)(?:\n#|\Z)",
        text_lower,
        re.DOTALL,
    )
    section = assessment_match.group(1) if assessment_match else text_lower

    # Check for explicit rejection signals
    reject_patterns = [
        r"\brequest\s*changes\b",
        r"\bchanges\s*requested\b",
        r"\breject(?:ed)?\b",
        r"\bdo\s*not\s*merge\b",
        r"\bblock(?:ing|ed)?\s*merge\b",
        r"\bnot\s*ready\s*(?:to\s*merge|for\s*merge)\b",
    ]
    for pattern in reject_patterns:
        if re.search(pattern, section):
            return "request_changes"

    # Check for explicit approval signals
    approve_patterns = [
        r"\bapprove[ds]?\b",
        r"\blgtm\b",
        r"\blooks\s*good\s*to\s*(?:me|merge)\b",
        r"\bready\s*to\s*merge\b",
        r"\bship\s*it\b",
    ]
    for pattern in approve_patterns:
        if re.search(pattern, section):
            return "approve"

    return "comment"


def set_commit_status(
    repo, sha: str, state: str, description: str, context: str
) -> None:
    """Set a commit status on the PR head SHA."""
    repo.get_commit(sha).create_status(
        state=state,
        description=description[:140],
        context=context,
    )
    print(f"Set commit status '{context}' to '{state}': {description}")


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

    # Get GitHub client first (needed to set status even when skipping)
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

    head_sha = pr.head.sha
    status_context = "ai-review/claude"

    # Configure Anthropic client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set, skipping Claude review")
        set_commit_status(
            repo, head_sha, "success",
            "Skipped - API key not configured", status_context,
        )
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Set pending status while review is running
    set_commit_status(
        repo, head_sha, "pending", "Claude review in progress...", status_context
    )

    # Get diff and files
    diff = get_pr_diff()
    changed_files = get_changed_files()

    if not diff:
        print("No diff found, skipping Claude review")
        set_commit_status(
            repo, head_sha, "success",
            "Skipped - no diff found", status_context,
        )
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
            model=os.environ.get("CLAUDE_REVIEW_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=2000,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            system=system_prompt,
        )

        review_text = response.content[0].text

        # Parse the verdict from the review
        verdict = parse_review_verdict(review_text)
        verdict_label = {
            "approve": "Approved",
            "request_changes": "Changes Requested",
            "comment": "Review Complete (no explicit approval)",
        }[verdict]

        # Post review comment
        review_body = f"""## Claude Code Review

{review_text}

---
**Sign-off status: {verdict_label}**
*Automated review by Anthropic Claude*
"""

        pr.create_issue_comment(review_body)
        print(f"Claude review posted successfully (verdict: {verdict})")

        # Set commit status based on verdict
        if verdict == "approve":
            set_commit_status(
                repo, head_sha, "success",
                "Claude approved this PR", status_context,
            )
        elif verdict == "request_changes":
            set_commit_status(
                repo, head_sha, "failure",
                "Claude requested changes", status_context,
            )
        else:
            # Comment without explicit approval — do not sign off
            set_commit_status(
                repo, head_sha, "failure",
                "Claude did not explicitly approve", status_context,
            )

    except anthropic.APIError as e:
        error_msg = f"Claude API error during review generation: {e}"
        print(error_msg)
        pr.create_issue_comment(
            f"Claude Code Review encountered an API error: {error_msg}"
        )
        set_commit_status(
            repo, head_sha, "error",
            "Claude review failed (API error)", status_context,
        )
    except Exception as e:
        error_msg = f"Claude review failed due to an unexpected error: {e}"
        print(error_msg)
        pr.create_issue_comment(
            f"Claude Code Review encountered an error: {error_msg}"
        )
        set_commit_status(
            repo, head_sha, "error",
            "Claude review failed (unexpected error)", status_context,
        )


if __name__ == "__main__":
    main()
