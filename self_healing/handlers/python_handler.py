"""
Python Handler for AI Analyst Self-Healing System.

Handles generic Python errors including syntax, imports, and runtime errors.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

from self_healing.models.errors import DetectedError, ErrorType, EnvironmentType, ErrorSeverity
from self_healing.models.fixes import Fix, FixStrategy, CodePatch


class PythonHandler:
    """
    Handles generic Python errors.

    Capabilities:
    - Syntax error fixes
    - Import management
    - Common runtime error patterns
    """

    def __init__(self) -> None:
        self._common_imports = {
            "pd": "import pandas as pd",
            "np": "import numpy as np",
            "plt": "import matplotlib.pyplot as plt",
            "sns": "import seaborn as sns",
            "json": "import json",
            "os": "import os",
            "sys": "import sys",
            "Path": "from pathlib import Path",
            "datetime": "from datetime import datetime",
            "Optional": "from typing import Optional",
            "List": "from typing import List",
            "Dict": "from typing import Dict",
            "Any": "from typing import Any",
        }

    def fix_syntax_error(
        self,
        error: DetectedError,
        file_path: Optional[Path] = None,
    ) -> Optional[Fix]:
        """Attempt to fix syntax errors."""
        if not file_path or not file_path.exists():
            return None

        content = file_path.read_text()
        line_num = error.location.line_number
        error_text = error.context.get("text", "")

        fixed_content = None
        description = ""

        # Common syntax fixes
        if "unexpected EOF" in error.message:
            fixed_content = self._fix_unclosed_brackets(content)
            description = "Close unclosed brackets"

        elif "invalid syntax" in error.message and ":" in error_text:
            fixed_content = self._fix_missing_colon(content, line_num)
            description = "Fix missing or misplaced colon"

        elif "expected ':'" in error.message:
            fixed_content = self._add_missing_colon(content, line_num)
            description = "Add missing colon"

        if not fixed_content:
            return None

        return Fix(
            error=error,
            strategy=FixStrategy.FIX_SYNTAX,
            description=description,
            reasoning=f"Syntax error at line {line_num}: {error.message}",
            code_patches=[
                CodePatch(
                    file_path=file_path,
                    original_content=content,
                    new_content=fixed_content,
                    line_start=line_num,
                    description=description,
                )
            ],
            confidence=0.65,
        )

    def _fix_unclosed_brackets(self, content: str) -> Optional[str]:
        """Fix unclosed brackets by adding closing ones."""
        brackets = {"(": ")", "[": "]", "{": "}"}
        stack = []

        for char in content:
            if char in brackets:
                stack.append(brackets[char])
            elif char in brackets.values():
                if stack and stack[-1] == char:
                    stack.pop()

        if stack:
            return content + "".join(reversed(stack))
        return None

    def _fix_missing_colon(self, content: str, line_num: int) -> Optional[str]:
        """Fix missing colon on function/class/control statements."""
        lines = content.split("\n")
        if line_num and line_num <= len(lines):
            line = lines[line_num - 1]
            if any(kw in line for kw in ["def ", "class ", "if ", "for ", "while ", "with ", "try", "except", "else", "elif"]):
                if not line.rstrip().endswith(":"):
                    lines[line_num - 1] = line.rstrip() + ":"
                    return "\n".join(lines)
        return None

    def _add_missing_colon(self, content: str, line_num: int) -> Optional[str]:
        """Add missing colon at end of line."""
        lines = content.split("\n")
        if line_num and line_num <= len(lines):
            line = lines[line_num - 1]
            if not line.rstrip().endswith(":"):
                lines[line_num - 1] = line.rstrip() + ":"
                return "\n".join(lines)
        return None

    def fix_import_error(
        self,
        error: DetectedError,
        file_path: Optional[Path] = None,
    ) -> Fix:
        """Generate fix for import errors."""
        module = error.context.get("module", "")

        # Check for alias usage
        for alias, import_statement in self._common_imports.items():
            if alias in module or module.split(".")[0] in import_statement:
                return Fix(
                    error=error,
                    strategy=FixStrategy.ADD_IMPORT,
                    description=f"Add missing import: {import_statement}",
                    reasoning=f"Module '{module}' not imported",
                    config_changes={"import_statement": import_statement},
                    confidence=0.90,
                )

        # Standard pip install fix
        package_name = module.split(".")[0]
        return Fix(
            error=error,
            strategy=FixStrategy.INSTALL_DEPENDENCY,
            description=f"Install missing package: {package_name}",
            reasoning=f"Module '{module}' not found",
            config_changes={"package": package_name},
            confidence=0.80,
        )

    def add_import_to_file(
        self,
        file_path: Path,
        import_statement: str,
    ) -> Optional[CodePatch]:
        """Add import statement to file."""
        if not file_path.exists():
            return None

        content = file_path.read_text()

        # Check if already imported
        if import_statement in content:
            return None

        # Find best location for import
        lines = content.split("\n")
        insert_line = 0

        for i, line in enumerate(lines):
            if line.startswith(("import ", "from ")):
                insert_line = i + 1
            elif line.startswith(('"""', "'''")):
                # Skip docstrings
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().endswith(('"""', "'''")):
                        insert_line = j + 1
                        break

        lines.insert(insert_line, import_statement)
        new_content = "\n".join(lines)

        return CodePatch(
            file_path=file_path,
            original_content=content,
            new_content=new_content,
            line_start=insert_line + 1,
            description=f"Add import: {import_statement}",
        )

    def fix_attribute_error(
        self,
        error: DetectedError,
    ) -> Fix:
        """Generate fix suggestions for attribute errors."""
        # Extract object and attribute from error message
        match = re.search(r"'(\w+)' object has no attribute '(\w+)'", error.message)

        suggestions = []
        if match:
            obj_type = match.group(1)
            attr = match.group(2)

            # Common typos
            common_fixes = {
                "lenght": "length",
                "heigth": "height",
                "widht": "width",
                "appedn": "append",
                "extnend": "extend",
            }

            if attr in common_fixes:
                suggestions.append(f"Did you mean '{common_fixes[attr]}'?")

            # DataFrame-specific suggestions
            if obj_type in ("DataFrame", "Series"):
                suggestions.append("Check available methods with dir(obj)")
                suggestions.append("Use df.columns to see column names")

        return Fix(
            error=error,
            strategy=FixStrategy.CODE_PATCH,
            description="Fix attribute error",
            reasoning=error.message,
            context={"suggestions": suggestions},
            confidence=0.50,
        )

    def fix_name_error(
        self,
        error: DetectedError,
        file_path: Optional[Path] = None,
    ) -> Fix:
        """Generate fix for undefined name errors."""
        match = re.search(r"name '(\w+)' is not defined", error.message)

        if match:
            name = match.group(1)

            # Check if it's a common import alias
            if name in self._common_imports:
                return Fix(
                    error=error,
                    strategy=FixStrategy.ADD_IMPORT,
                    description=f"Add missing import for '{name}'",
                    reasoning=f"'{name}' used but not imported",
                    config_changes={"import_statement": self._common_imports[name]},
                    confidence=0.90,
                )

        return Fix(
            error=error,
            strategy=FixStrategy.MANUAL,
            description="Define or import missing name",
            reasoning=error.message,
            confidence=0.30,
        )
