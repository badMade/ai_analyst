#!/usr/bin/env python3
"""
Gemini Code Assist - Automated code review for AI-created branches.
Reviews pull requests from branches created by AI assistants (claude/, codex/, jules/, copilot/).
"""

import os
import subprocess

import google.generativeai as genai
from github import Github

# Maximum length for diff content before truncation (Gemini token limits)
MAX_DIFF_LENGTH = 30000
MAX_TITLE_LENGTH = 256
MAX_BODY_LENGTH = 4000


def get_pr_diff() -> str:
    """Get the diff for the current PR."""
    # First try to read from file (set by workflow)
    if os.path.exists("/tmp/pr_diff.txt"):
        try:
            with open("/tmp/pr_diff.txt") as f:
                file_diff = f.read()
        except OSError as e:
            print(f"Error reading PR diff file: {e}")
            file_diff = ""

        # If the file contains a non-empty diff, use it; otherwise fall back to git diff
        if file_diff.strip():
            return file_diff
    # Fallback to git diff
    try:
        base_ref = os.environ.get("GITHUB_BASE_REF", "main")
        result = subprocess.run(
            ["git", "diff", f"origin/{base_ref}...HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        return result.stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Error getting PR diff: {e}")
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
            timeout=60,
        )
        return [f for f in result.stdout.strip().split("\n") if f]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Error getting changed files: {e}")
        return []


def get_ai_assistant_name(head_ref: str) -> str:
    """Determine which AI assistant created the branch."""
    assistant_map = {
        "claude/": "Claude",
        "codex/": "OpenAI Codex",
        "jules/": "Google Jules",
        "copilot/": "GitHub Copilot",
    }
    for prefix, name in assistant_map.items():
        if head_ref.startswith(prefix):
            return name
    return "AI Assistant"


def main():
    # Configure Gemini
    api_key = os.environ.get("GOOGLE_AI_API_KEY")
    if not api_key:
        print("GOOGLE_AI_API_KEY not set, skipping Gemini Code Assist review")
        return

    genai.configure(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    model = genai.GenerativeModel(model_name)

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

    pr_number_str = os.environ.get("PR_NUMBER")
    if not pr_number_str:
        print("PR_NUMBER not set")
        return

    try:
        pr = repo.get_pull(int(pr_number_str))
    except ValueError:
        print(f"PR_NUMBER '{pr_number_str}' is not a valid integer.")
        return

    # Get branch info
    head_ref = os.environ.get("HEAD_REF", pr.head.ref)
    ai_assistant = get_ai_assistant_name(head_ref)

    # Get diff and files
    diff = get_pr_diff()
    changed_files = get_changed_files()

    if not diff:
        print("No diff found")
        return

    # Truncate diff if too long (Gemini has token limits)
    if len(diff) > MAX_DIFF_LENGTH:
        diff = diff[:MAX_DIFF_LENGTH] + "\n... (diff truncated)"

    # Sanitize PR metadata to avoid excessively long or malformed inputs
    max_title_length = 256
    max_body_length = 4000

    pr_title = (pr.title or "").strip()
    if len(pr_title) > max_title_length:
        pr_title = pr_title[:max_title_length] + "…"

    pr_body = (pr.body or "No description provided").strip()
    if len(pr_body) > max_body_length:
        pr_body = pr_body[:max_body_length] + "\n… (description truncated)"

    # Build review prompt tailored for AI-generated code
    prompt = f"""You are Gemini Code Assist, an expert code reviewer. This pull request was created by {ai_assistant}.
Review the AI-generated code changes and provide constructive feedback.

## Pull Request
**Title:** {pr_title}
**Description:** {pr_body}
**Source Branch:** {head_ref}
**Created by:** {ai_assistant}

## Changed Files
{'\n'.join(f'- {f}' for f in changed_files if f)}

## Diff
```diff
{diff}
```

## Review Instructions
As Gemini Code Assist, provide a thorough review of this AI-generated code:

1. **Summary**: Brief overview of what this PR accomplishes
2. **Code Quality**: Evaluate readability, maintainability, and adherence to best practices
3. **Correctness**: Verify the implementation is correct and handles edge cases
4. **Security**: Check for any security vulnerabilities or concerns
5. **Suggestions**: Specific improvements with code examples where helpful
6. **AI-Generated Code Notes**: Any patterns typical of AI-generated code that should be reviewed
7. **Overall Assessment**: Approve, request changes, or comment

Format your response in GitHub-flavored markdown. Be constructive and specific.
Focus on issues that AI assistants commonly miss, such as:
- Error handling edge cases
- Input validation
- Resource cleanup
- Documentation accuracy
- Test coverage gaps"""

    try:
        # Generate review
        response = model.generate_content(prompt)
        review_text = getattr(response, "text", None)

        # Validate generated review content before posting
        if review_text is None or not str(review_text).strip():
            fallback_message = (
                f"@gemini-code-assist was unable to generate a review for this {ai_assistant}-created PR "
                "because the model returned no usable content. This can happen due to rate limiting, "
                "temporary service issues, or safety filters blocking the response. "
                "Please try rerunning the workflow or perform a manual review."
            )
            pr.create_issue_comment(fallback_message)
            print("Gemini Code Assist could not generate review: empty or missing response.text")
            return
        # Post review comment
        review_body = f"""## @gemini-code-assist Review

*Reviewing PR from `{head_ref}` (created by {ai_assistant})*

{review_text}

---
*Automated review by Gemini Code Assist*
"""

        pr.create_issue_comment(review_body)
        print(f"Gemini Code Assist review posted successfully for {ai_assistant}-created branch")

    except Exception as e:
        print(f"Error generating review: {e}")
        error_comment = (
            f"@gemini-code-assist encountered an error reviewing this {ai_assistant}-created PR. "
            "Please check the workflow logs for details."
        )
        try:
            pr.create_issue_comment(error_comment)
        except Exception as comment_error:
            print(f"Failed to post error comment to PR: {comment_error}")


if __name__ == "__main__":
    main()
