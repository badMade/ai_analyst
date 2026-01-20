#!/usr/bin/env python3
"""
Gemini AI Studio Assistant for GitHub Issues and PRs.
Responds to @gemini mentions with AI-powered assistance.
"""

import json
import os
import re

import google.generativeai as genai
from github import Github


def get_event_data():
    """Load GitHub event data."""
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and os.path.exists(event_path):
        with open(event_path) as f:
            return json.load(f)
    return {}


def extract_prompt(text: str) -> str:
    """Extract the prompt after @gemini mention."""
    match = re.search(r"@gemini\s+(.+)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def main():
    # Configure Gemini
    api_key = os.environ.get("GOOGLE_AI_API_KEY")
    if not api_key:
        print("GOOGLE_AI_API_KEY not set")
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
        print("No prompt found after @gemini mention")
        return

    # Build context
    context = f"""You are Gemini, an AI assistant helping with a GitHub repository.

Repository: {repo.full_name}
Title: {issue_or_pr.title}
Description: {issue_or_pr.body or 'No description'}

User request: {prompt}

Provide a helpful, detailed response. If this is about code, provide specific suggestions.
Format your response in GitHub-flavored markdown."""

    # Generate response
    try:
        response = model.generate_content(context)
        reply = response.text

        # Post response
        issue_or_pr.create_comment(f"**Gemini AI Studio Response:**\n\n{reply}")
        print("Response posted successfully")

    except Exception as e:
        error_msg = "Gemini encountered an error. Please check the workflow logs for details."
        issue_or_pr.create_comment(error_msg)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
