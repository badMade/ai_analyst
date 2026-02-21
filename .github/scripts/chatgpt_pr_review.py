#!/usr/bin/env python3
"""
ChatGPT/Codex PR Review - Automated code review using OpenAI.
Analyzes pull requests and provides detailed feedback.
Sets a commit status to block or approve auto-merge.
"""

import os
import re
import subprocess

from github import Github
from openai import OpenAI


def get_pr_diff() -> str:
    """Get the diff for the current PR."""
    try:
        with open("/tmp/pr_diff.txt") as f:
            return f.read()
    except FileNotFoundError:
        # Fail loudly if the diff file is missing so workflow issues are visible
        raise FileNotFoundError(
            "PR diff file '/tmp/pr_diff.txt' not found. "
            "Ensure the previous workflow step generated this file."
        )


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
    status_context = "ai-review/chatgpt"

    # Configure OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping ChatGPT review")
        set_commit_status(
            repo,
            head_sha,
            "success",
            "Skipped - API key not configured",
            status_context,
        )
        return

    client = OpenAI(api_key=api_key)

    # Set pending status while review is running
    set_commit_status(repo, head_sha, "pending", "ChatGPT review in progress...", status_context)

    # Get diff and context
    diff = get_pr_diff()
    changed_files = get_changed_files()
    pr_title = os.environ.get("PR_TITLE", pr.title)
    pr_body = os.environ.get("PR_BODY", pr.body) or "No description"

    if not diff:
        print("No diff found, skipping ChatGPT review")
        set_commit_status(
            repo,
            head_sha,
            "success",
            "Skipped - no diff found",
            status_context,
        )
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
{chr(10).join(f"- {f}" for f in changed_files)}

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

        # Parse the verdict from the review
        verdict = parse_review_verdict(review_text)
        verdict_label = {
            "approve": "Approved",
            "request_changes": "Changes Requested",
            "comment": "Review Complete (no explicit approval)",
        }[verdict]

        # Post review comment
        review_body = f"""## ChatGPT Codex Review

{review_text}

---
**Sign-off status: {verdict_label}**
*Automated review by OpenAI ChatGPT/Codex*
"""

        pr.create_issue_comment(review_body)
        print(f"ChatGPT review posted successfully (verdict: {verdict})")

        # Set commit status based on verdict
        if verdict == "approve":
            set_commit_status(
                repo,
                head_sha,
                "success",
                "ChatGPT approved this PR",
                status_context,
            )
        elif verdict == "request_changes":
            set_commit_status(
                repo,
                head_sha,
                "failure",
                "ChatGPT requested changes",
                status_context,
            )
        else:
            set_commit_status(
                repo,
                head_sha,
                "failure",
                "ChatGPT did not explicitly approve",
                status_context,
            )

    except Exception as e:
        print(f"Error generating review: {e}")
        pr.create_issue_comment(f"ChatGPT Code Review encountered an error: {str(e)}")
        set_commit_status(
            repo,
            head_sha,
            "error",
            "ChatGPT review failed",
            status_context,
        )


if __name__ == "__main__":
    main()
