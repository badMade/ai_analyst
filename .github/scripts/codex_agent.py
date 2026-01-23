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
    match = re.search(r"@codex-agent(?:\s+(.+))?", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    prompt = (match.group(1) or "").strip()
    return prompt


def get_file_content(file_path: str) -> str:
    """Read file content."""
    try:
        with open(file_path) as f:
            return f.read()
    except Exception:
        return ""


def write_file_content(file_path: str, content: str) -> bool:
    """Write content to file with path validation to prevent directory traversal."""
    try:
        # Get workspace root for path validation
        workspace_root = os.path.realpath(os.environ.get("GITHUB_WORKSPACE", os.getcwd()))

        # Resolve the target path, resolving symlinks
        if os.path.isabs(file_path):
            target_path = os.path.realpath(file_path)
        else:
            target_path = os.path.realpath(os.path.join(workspace_root, file_path))

        # Validate path is within workspace to prevent directory traversal
        if not (target_path == workspace_root or target_path.startswith(workspace_root + os.sep)):
            print(f"Security: Refusing to write outside workspace: {file_path}")
            return False

        dir_path = os.path.dirname(target_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(target_path, "w") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error writing file {file_path}: {e}")
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


def resolve_issue_or_pr(repo: Github, event: dict) -> tuple:
    """Resolve the issue or PR object and comment body from the event."""
    comment_body = ""
    issue_or_pr = None

    if "comment" in event:
        comment_body = event["comment"].get("body", "")
        if "issue" in event:
            issue_or_pr = repo.get_issue(event["issue"]["number"])
        elif "pull_request" in event:
            issue_or_pr = repo.get_pull(event["pull_request"]["number"])
    elif "review" in event:
        comment_body = event["review"].get("body", "")
        if "pull_request" in event:
            issue_or_pr = repo.get_pull(event["pull_request"]["number"])
    elif "issue" in event:
        comment_body = event["issue"].get("body", "")
        issue_or_pr = repo.get_issue(event["issue"]["number"])

    return issue_or_pr, comment_body


def get_author_association(event: dict) -> str:
    """Return the author association if available."""
    for key in ("comment", "issue", "review"):
        association = event.get(key, {}).get("author_association")
        if association:
            return association
    return ""


def list_repo_files(max_files: int) -> list[str]:
    """List tracked repository files for context."""
    file_list = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=False,
    )
    files = [f for f in file_list.stdout.splitlines() if f]
    allowed_extensions = tuple(e.strip() for e in os.environ.get("CODEX_AGENT_ALLOWED_EXTENSIONS", ".py,.js,.ts,.md,.yml,.yaml,.toml,.json").split(","))
    filtered_files = [f for f in files if f.endswith(allowed_extensions)]
    return filtered_files[:max_files]


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
    issue_or_pr, comment_body = resolve_issue_or_pr(repo, event)

    if not issue_or_pr:
        print("Could not determine issue or PR context")
        return

    # Validate author association as a defense-in-depth check
    author_association = get_author_association(event)
    allowed_associations = {"OWNER", "MEMBER", "COLLABORATOR"}
    if author_association and author_association not in allowed_associations:
        print(f"Unauthorized author association: {author_association}")
        return

    # Extract prompt
    prompt = extract_prompt(comment_body)
    if not prompt:
        print("No prompt found after @codex-agent mention")
        return

    # Get repository file structure
    try:
        max_files_to_show = int(os.environ.get("CODEX_AGENT_MAX_FILES", "50"))
    except ValueError:
        max_files_to_show = 50
    max_files_to_show = max(1, max_files_to_show)
    files = list_repo_files(max_files_to_show)

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

    files_for_context = files

    user_message = f"""Repository: {repo.full_name}
Issue/PR: {issue_or_pr.title}
Description: {issue_or_pr.body or 'No description'}

Files in repository:
{chr(10).join(files_for_context)}

User request: {prompt}

Analyze the request and provide changes if needed."""

    try:
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
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
            workspace_root = os.path.abspath(os.environ.get("GITHUB_WORKSPACE", os.getcwd()))
            for change in changes:
                # Resolve the target path and ensure it stays within the workspace root
                file_spec = change.get("file", "")
                if not file_spec:
                    print("Skipping change with empty file path")
                    continue

                if os.path.isabs(file_spec):
                    target_path = os.path.abspath(file_spec)
                else:
                    target_path = os.path.abspath(os.path.join(workspace_root, file_spec))

                if not (target_path == workspace_root or target_path.startswith(workspace_root + os.sep)):
                    print(f"Skipping unsafe file operation outside workspace: {file_spec} -> {target_path}")
                    continue

                if change["action"] in ["modify", "create"]:
                    # write_file_content is expected to work with repository-relative paths
                    write_file_content(file_spec, change["content"])
                elif change["action"] == "delete":
                    if os.path.exists(target_path):
                        os.remove(target_path)

            # Create branch and commit
            branch_name = f"codex-agent/{issue_or_pr.number}"
            try:
                # Try to create a new branch and switch to it
                subprocess.run(["git", "checkout", "-b", branch_name], check=True)
            except subprocess.CalledProcessError:
                # If the branch already exists, switch to it instead
                subprocess.run(["git", "checkout", branch_name], check=True)

            if git_commit_and_push(commit_msg, branch_name):
                reply += f"\n\nChanges committed to branch `{branch_name}`"
            else:
                reply += "\n\n*Note: Could not commit changes automatically*"

        issue_or_pr.create_comment(reply)
        print("Codex Agent completed successfully")

    except Exception as e:
        error_msg = f"Codex Agent encountered an error: {str(e)}"
        issue_or_pr.create_comment(error_msg)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
