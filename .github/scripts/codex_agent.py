#!/usr/bin/env python3
"""
OpenAI Codex Agent for GitHub.
Responds to @codex-agent mentions and can make code changes.
"""

import json
import os
import re
import subprocess

from github import Github
from openai import OpenAI


def get_event_data():
    """Load GitHub event data."""
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and os.path.exists(event_path):
        with open(event_path) as f:
            return json.load(f)
    return {}


def extract_prompt(text: str) -> str:
    """Extract the prompt after @codex-agent mention."""
    match = re.search(r"@codex-agent\s+(.+)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def get_file_content(file_path: str) -> str:
    """Read file content."""
    try:
        with open(file_path) as f:
            return f.read()
    except Exception:
        return ""


def write_file_content(file_path: str, content: str) -> bool:
    """Write content to file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)
        return True
    except Exception:
        return False


def git_commit_and_push(message: str, branch: str) -> bool:
    """Commit changes and push to branch."""
    try:
        subprocess.run(["git", "config", "user.email", "codex-agent@github.com"], check=True)
        subprocess.run(["git", "config", "user.name", "Codex Agent"], check=True)
        subprocess.run(["git", "add", "-A"], check=True)
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "push", "origin", f"HEAD:{branch}"], check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    # Configure OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set")
        return

    client = OpenAI(api_key=api_key)

    # Get GitHub client
    gh_token = os.environ.get("GITHUB_TOKEN")
    if not gh_token:
        print("GITHUB_TOKEN not set")
        return

    gh = Github(gh_token)
    repo = gh.get_repo(os.environ.get("GITHUB_REPOSITORY"))

    # Load event data
    event = get_event_data()

    # Determine context
    comment_body = ""
    issue_or_pr = None
    is_pr = False

    if "comment" in event:
        comment_body = event["comment"].get("body", "")
        if "issue" in event:
            issue_or_pr = repo.get_issue(event["issue"]["number"])
        elif "pull_request" in event:
            issue_or_pr = repo.get_pull(event["pull_request"]["number"])
            is_pr = True
    elif "issue" in event:
        comment_body = event["issue"].get("body", "")
        issue_or_pr = repo.get_issue(event["issue"]["number"])

    if not issue_or_pr:
        print("Could not determine issue or PR context")
        return

    # Extract prompt
    prompt = extract_prompt(comment_body)
    if not prompt:
        print("No prompt found after @codex-agent mention")
        return

    # Get repository file structure
    file_list = subprocess.run(
        ["find", ".", "-type", "f", "-name", "*.py", "-o", "-name", "*.js", "-o", "-name", "*.ts"],
        capture_output=True,
        text=True,
    )
    files = [f for f in file_list.stdout.strip().split("\n") if f and not f.startswith("./.git")]

    # Build agent prompt
    system_message = """You are Codex Agent, an AI coding assistant that can analyze and modify code.
When asked to make changes, provide your response in this JSON format:
{
    "analysis": "Your analysis of what needs to be done",
    "changes": [
        {
            "file": "path/to/file.py",
            "action": "modify|create|delete",
            "content": "full file content if modify/create"
        }
    ],
    "commit_message": "Description of changes made"
}

If no code changes are needed, set "changes" to an empty array and provide your response in "analysis"."""

    user_message = f"""Repository: {repo.full_name}
Issue/PR: {issue_or_pr.title}
Description: {issue_or_pr.body or 'No description'}

Files in repository:
{chr(10).join(files[:50])}

User request: {prompt}

Analyze the request and provide changes if needed."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=3000,
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        analysis = result.get("analysis", "")
        changes = result.get("changes", [])
        commit_msg = result.get("commit_message", "Codex Agent changes")

        # Post analysis
        reply = f"**Codex Agent Analysis:**\n\n{analysis}\n\n"

        if changes:
            reply += f"**Proposed Changes ({len(changes)} files):**\n"
            for change in changes:
                reply += f"- `{change['file']}` ({change['action']})\n"

            # Apply changes
            for change in changes:
                if change["action"] in ["modify", "create"]:
                    write_file_content(change["file"], change["content"])
                elif change["action"] == "delete":
                    if os.path.exists(change["file"]):
                        os.remove(change["file"])

            # Create branch and commit
            branch_name = f"codex-agent/{issue_or_pr.number}"
            # Try to create and check out a new branch. If that fails (e.g., branch exists),
            # fall back to checking out the existing branch. If both fail, do not commit.
            create_branch_result = subprocess.run(
                ["git", "checkout", "-b", branch_name],
                check=False,
                capture_output=True,
                text=True,
            )
            if create_branch_result.returncode != 0:
                checkout_existing_result = subprocess.run(
                    ["git", "checkout", branch_name],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if checkout_existing_result.returncode != 0:
                    reply += (
                        "\n\n*Note: Failed to create or checkout branch "
                        f"`{branch_name}`; changes were not committed automatically.*"
                    )
                    issue_or_pr.create_comment(reply)
                    print(
                        f"Failed to create or checkout branch {branch_name}: "
                        f"{create_branch_result.stderr or checkout_existing_result.stderr}"
                    )
                    return

            if git_commit_and_push(commit_msg, branch_name):
                reply += f"\n\nChanges committed to branch `{branch_name}`"
            else:
                reply += "\n\n*Note: Could not commit changes automatically*"

        issue_or_pr.create_comment(reply)
        print("Codex Agent completed successfully")

    except Exception as e:
        error_msg = "Codex Agent encountered an error. Please check the workflow logs for details."
        issue_or_pr.create_comment(error_msg)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
