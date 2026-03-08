"""
CI Handler for AI Analyst Self-Healing System.

Handles CI/CD-specific errors including GitHub Actions, workflow failures,
and test failures.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from self_healing.models.errors import DetectedError, ErrorType, EnvironmentType, ErrorSeverity
from self_healing.models.fixes import Fix, FixStrategy, CodePatch


class CIHandler:
    """
    Handles CI/CD-specific errors.

    Capabilities:
    - Parse GitHub Actions logs
    - Detect workflow failures
    - Generate fixes for common CI issues
    - Handle test failures
    """

    def __init__(self) -> None:
        self._common_ci_errors = {
            "Permission denied": FixStrategy.SET_PERMISSION,
            "No such file or directory": FixStrategy.CREATE_DIRECTORY,
            "ModuleNotFoundError": FixStrategy.INSTALL_DEPENDENCY,
            "SyntaxError": FixStrategy.FIX_SYNTAX,
            "rate limit": FixStrategy.RETRY_CI,
        }

    def detect_ci_errors(self, log_content: str) -> list[DetectedError]:
        """Detect errors from CI log content."""
        errors = []

        # Check for workflow failure
        if re.search(r"(workflow|job|step).*failed", log_content, re.IGNORECASE):
            errors.append(DetectedError(
                error_type=ErrorType.CI_FAILURE,
                severity=ErrorSeverity.ERROR,
                environment=EnvironmentType.CI,
                message=self._extract_failure_message(log_content),
                context={"log_excerpt": log_content[:1000]},
            ))

        # Check for test failures
        test_failures = re.findall(r"FAILED\s+(\S+)", log_content)
        if test_failures:
            errors.append(DetectedError(
                error_type=ErrorType.TEST_FAILURE,
                severity=ErrorSeverity.ERROR,
                environment=EnvironmentType.CI,
                message=f"Tests failed: {', '.join(test_failures[:5])}",
                context={"failed_tests": test_failures},
            ))

        # Check for lint errors
        if re.search(r"(flake8|pylint|mypy|ruff).*error", log_content, re.IGNORECASE):
            errors.append(DetectedError(
                error_type=ErrorType.LINT_ERROR,
                severity=ErrorSeverity.WARNING,
                environment=EnvironmentType.CI,
                message="Linting errors detected",
                context={"log_excerpt": log_content[:500]},
            ))

        return errors

    def _extract_failure_message(self, log_content: str) -> str:
        """Extract the main failure message from CI logs."""
        # Look for error lines
        error_patterns = [
            r"Error:\s*(.+)",
            r"fatal:\s*(.+)",
            r"FAILED\s*(.+)",
        ]

        for pattern in error_patterns:
            match = re.search(pattern, log_content, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:200]

        return "CI workflow failed"

    def generate_workflow_fix(
        self,
        error: DetectedError,
        workflow_path: Optional[Path] = None,
    ) -> Fix:
        """Generate fix for workflow errors."""
        message = error.message.lower()

        # Determine fix strategy
        for pattern, strategy in self._common_ci_errors.items():
            if pattern.lower() in message:
                return self._create_fix_for_strategy(error, strategy, workflow_path)

        # Default to retry
        return Fix(
            error=error,
            strategy=FixStrategy.RETRY_CI,
            description="Retry failed CI workflow",
            reasoning="Workflow failure detected, attempting retry",
            confidence=0.50,
        )

    def _create_fix_for_strategy(
        self,
        error: DetectedError,
        strategy: FixStrategy,
        workflow_path: Optional[Path],
    ) -> Fix:
        """Create fix based on strategy."""
        if strategy == FixStrategy.SET_PERMISSION:
            return Fix(
                error=error,
                strategy=strategy,
                description="Add execute permissions to scripts",
                reasoning="Permission denied error in CI",
                config_changes={"chmod": "+x"},
                confidence=0.80,
            )

        if strategy == FixStrategy.CREATE_DIRECTORY:
            return Fix(
                error=error,
                strategy=strategy,
                description="Ensure required directories exist",
                reasoning="Missing directory in CI environment",
                config_changes={"mkdir_p": True},
                confidence=0.75,
            )

        if strategy == FixStrategy.INSTALL_DEPENDENCY:
            # Extract module name from error
            module = self._extract_module_name(error.message)
            return Fix(
                error=error,
                strategy=strategy,
                description=f"Add missing dependency: {module}",
                reasoning=f"Module {module} not found in CI",
                config_changes={"package": module},
                confidence=0.85,
            )

        return Fix(
            error=error,
            strategy=strategy,
            description=f"Apply {strategy.value} fix",
            reasoning=error.message,
            confidence=0.60,
        )

    def _extract_module_name(self, message: str) -> str:
        """Extract module name from import error message."""
        match = re.search(r"No module named ['\"]?(\w+)", message)
        if match:
            return match.group(1)
        return "unknown"

    def fix_workflow_syntax(
        self,
        workflow_path: Path,
    ) -> Optional[Fix]:
        """Fix common YAML syntax issues in workflows."""
        if not workflow_path.exists():
            return None

        content = workflow_path.read_text()
        original = content
        fixed = False

        # Fix common issues
        fixes_applied = []

        # Fix missing 'on' trigger
        if "on:" not in content:
            content = content.replace(
                "name:",
                "on: [push, pull_request]\n\nname:",
                1,
            )
            fixed = True
            fixes_applied.append("Added missing 'on' trigger")

        # Fix tabs (YAML doesn't allow tabs)
        if "\t" in content:
            content = content.replace("\t", "  ")
            fixed = True
            fixes_applied.append("Replaced tabs with spaces")

        if not fixed:
            return None

        return Fix(
            error=DetectedError(
                error_type=ErrorType.CI_FAILURE,
                severity=ErrorSeverity.ERROR,
                environment=EnvironmentType.CI,
                message="Workflow syntax issues",
            ),
            strategy=FixStrategy.FIX_WORKFLOW,
            description="Fix workflow YAML syntax",
            reasoning=f"Applied fixes: {', '.join(fixes_applied)}",
            code_patches=[
                CodePatch(
                    file_path=workflow_path,
                    original_content=original,
                    new_content=content,
                    description="Workflow syntax fix",
                )
            ],
            confidence=0.85,
        )

    def generate_missing_workflow_step(
        self,
        step_type: str,
    ) -> str:
        """Generate YAML for common missing workflow steps."""
        steps = {
            "checkout": """
      - name: Checkout code
        uses: actions/checkout@v4
""",
            "python_setup": """
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
""",
            "install_deps": """
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
""",
            "run_tests": """
      - name: Run tests
        run: pytest
""",
            "lint": """
      - name: Lint with ruff
        run: |
          pip install ruff
          ruff check .
""",
        }

        return steps.get(step_type, "")
