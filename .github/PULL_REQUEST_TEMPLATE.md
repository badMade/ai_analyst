## Description

<!-- Describe your changes here -->

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

<!-- How was this tested? -->

- [ ] Unit tests added/updated
- [ ] Manual testing performed

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated (if needed)
- [ ] No sensitive data exposed

---

## AI Code Review Integration

This repository uses multiple AI assistants for code review and assistance:

### Automatic Reviews
All PRs automatically receive reviews from:
- **Claude Code** - Anthropic's AI code review
- **Gemini Code Assist** - Google's enterprise code review
- **ChatGPT Codex** - OpenAI's GPT-4o code review

### On-Demand Assistance
Use these triggers in comments:

| Trigger | AI Assistant | Provider | Description |
|---------|--------------|----------|-------------|
| `@claude` | Claude Code | Anthropic | AI code assistance |
| `@claude agent` | Claude Agent | Anthropic | Automated code changes |
| `@gemini` | Gemini AI Studio | Google | AI for questions and help |
| `@jules` | Google Labs Jules | Google | AI coding agent |
| `@chatgpt` | ChatGPT | OpenAI | GPT-4o assistance |
| `@codex` | Codex | OpenAI | Code-focused help |
| `@codex-agent` | Codex Agent | OpenAI | Automated code changes |

### Example Usage
```
@gemini Can you explain what this function does?
@jules Please refactor this to use async/await
@claude Review the error handling in this PR
@chatgpt What's the best way to handle this edge case?
@codex-agent Add unit tests for this function
```
