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
- **Gemini Code Assist** - Google's enterprise code review
- **Gemini AI Studio** - Detailed PR analysis

### On-Demand Assistance
Use these triggers in comments:

| Trigger | AI Assistant | Description |
|---------|--------------|-------------|
| `@claude` | Claude Code | Anthropic's AI for code assistance |
| `@claude agent` | Claude Agent | Automated code changes |
| `@gemini` | Gemini AI Studio | Google AI for questions and help |
| `@jules` | Google Labs Jules | Google's AI coding agent |

### Example Usage
```
@gemini Can you explain what this function does?
@jules Please refactor this to use async/await
@claude Review the error handling in this PR
```
