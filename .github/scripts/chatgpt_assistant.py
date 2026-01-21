#!/usr/bin/env python3
"""
ChatGPT Assistant for GitHub Issues and PRs.
Responds to @chatgpt and @codex mentions with AI-powered assistance.
"""

import json
import os
import re

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
    """Extract the prompt after @chatgpt or @codex mention."""
    # Try @chatgpt first, then @codex
    for trigger in ["@chatgpt", "@codex"]:
        pattern = rf"{trigger}\s+(.+)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


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

    if "comment" in event:
        comment_body = event["comment"].get("body", "")
        if "issue" in event:
            issue_or_pr = repo.get_issue(event["issue"]["number"])
        elif "pull_request" in event:
            issue_or_pr = repo.get_pull(event["pull_request"]["number"])
    elif "issue" in event:
        comment_body = event["issue"].get("body", "")
        issue_or_pr = repo.get_issue(event["issue"]["number"])

    if not issue_or_pr:
        print("Could not determine issue or PR context")
        return

    # Extract prompt
    prompt = extract_prompt(comment_body)
    if not prompt:
        print("No prompt found after @chatgpt or @codex mention")
        return

    # Build context
    system_message = """You are ChatGPT, an AI assistant helping with a GitHub repository.
You provide helpful, accurate, and detailed responses about code and development.
Format responses in GitHub-flavored markdown."""

    user_message = f"""Repository: {repo.full_name}
Title: {issue_or_pr.title}
Description: {issue_or_pr.body or 'No description'}

User request: {prompt}

Provide a helpful, detailed response."""

    # Generate response
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1500,
            temperature=0.7,
        )

        reply = response.choices[0].message.content

        # Post response
        issue_or_pr.create_comment(f"**ChatGPT Response:**\n\n{reply}")
        print("Response posted successfully")

    except Exception as e:
        print(f"Error: {e}")
        issue_or_pr.create_comment(
            "ChatGPT encountered an error. Please check the workflow logs for details."
        )


if __name__ == "__main__":
    main()
