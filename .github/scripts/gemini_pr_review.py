#!/usr/bin/env python3
"""
Gemini PR Review - Automated code review using Google AI Studio.
Analyzes pull requests and provides detailed feedback.
Sets a commit status to block or approve auto-merge.
"""

import os
import re
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


def set_commit_status(repo, sha: str, state: str, description: str, context: str) -> None:
    """Set a commit status on the PR head SHA."""
    repo.get_commit(sha).create_status(
        state=state,
        description=description[:140],
        context=context,
    )
    print(f"Set commit status '{context}' to '{state}': {description}")


def main():
    # Get GitHub client first (needed to set status even when skipping)
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

    head_sha = pr.head.sha
    status_context = "ai-review/gemini"

    # Configure Gemini
    api_key = os.environ.get("GOOGLE_AI_API_KEY")
    if not api_key:
        print("GOOGLE_AI_API_KEY not set, skipping Gemini review")
        set_commit_status(
            repo,
            head_sha,
            "success",
            "Skipped - API key not configured",
            status_context,
        )
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Set pending status while review is running
    set_commit_status(repo, head_sha, "pending", "Gemini review in progress...", status_context)

    # Get diff and files
    diff = get_pr_diff()
    changed_files = get_changed_files()

    if not diff:
        print("No diff found, skipping Gemini review")
        set_commit_status(
            repo,
            head_sha,
            "success",
            "Skipped - no diff found",
            status_context,
        )
        return

    # Truncate diff if too long (Gemini has token limits)
    max_diff_length = 30000
    if len(diff) > max_diff_length:
        diff = diff[:max_diff_length] + "\n... (diff truncated)"

    # Build review prompt
    prompt = f"""You are an expert code reviewer. Review this pull request and provide constructive feedback.

## Pull Request
**Title:** {pr.title}
**Description:** {pr.body or "No description provided"}

## Changed Files
{chr(10).join(f"- {f}" for f in changed_files if f)}

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
5. **Implementation Details**: For your suggestions, what are the steps, files, and line numbers required to implement them?
6. **Overall Assessment**: Approve, request changes, or comment

Format your response in GitHub-flavored markdown. Be constructive and specific.
Use inline code references where applicable."""

    try:
        # Generate review
        response = model.generate_content(prompt)
        review_text = response.text

        # Parse the verdict from the review
        verdict = parse_review_verdict(review_text)
        verdict_label = {
            "approve": "Approved",
            "request_changes": "Changes Requested",
            "comment": "Review Complete (no explicit approval)",
        }[verdict]

        # Post review comment
        review_body = f"""## Gemini Code Review

{review_text}

---
**Sign-off status: {verdict_label}**
*Automated review by Google Gemini AI Studio*
"""

        pr.create_issue_comment(review_body)
        print(f"Gemini review posted successfully (verdict: {verdict})")

        # Set commit status based on verdict
        if verdict == "approve":
            set_commit_status(
                repo,
                head_sha,
                "success",
                "Gemini approved this PR",
                status_context,
            )
        elif verdict == "request_changes":
            set_commit_status(
                repo,
                head_sha,
                "failure",
                "Gemini requested changes",
                status_context,
            )
        else:
            set_commit_status(
                repo,
                head_sha,
                "failure",
                "Gemini did not explicitly approve",
                status_context,
            )

    except Exception as e:
        print(f"Error generating review: {e}")
        pr.create_issue_comment(f"Gemini Code Review encountered an error: {str(e)}")
        set_commit_status(
            repo,
            head_sha,
            "error",
            "Gemini review failed",
            status_context,
        )


if __name__ == "__main__":
    main()
