"""Validator utility for Agentic AI.

Provides data validation and integrity checking.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Type
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationError:
    """A validation error or warning."""
    field: str
    message: str
    level: ValidationLevel = ValidationLevel.ERROR
    value: Any = None


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    def add_error(self, field: str, message: str, value: Any = None) -> None:
        """Add an error."""
        self.errors.append(ValidationError(
            field=field,
            message=message,
            level=ValidationLevel.ERROR,
            value=value,
        ))
        self.is_valid = False

    def add_warning(self, field: str, message: str, value: Any = None) -> None:
        """Add a warning."""
        self.warnings.append(ValidationError(
            field=field,
            message=message,
            level=ValidationLevel.WARNING,
            value=value,
        ))

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another validation result."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.is_valid = self.is_valid and other.is_valid
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": [
                {"field": e.field, "message": e.message}
                for e in self.errors
            ],
            "warnings": [
                {"field": w.field, "message": w.message}
                for w in self.warnings
            ],
        }


class Validator:
    """Data validator with configurable rules."""

    def __init__(self):
        """Initialize the validator."""
        self._rules: dict[str, list[Callable]] = {}
        self._type_validators: dict[Type, Callable] = {}

    def add_rule(
        self,
        field: str,
        rule: Callable[[Any], tuple[bool, str]],
    ) -> None:
        """Add a validation rule for a field.

        Args:
            field: Field name.
            rule: Validation function returning (is_valid, message).
        """
        if field not in self._rules:
            self._rules[field] = []
        self._rules[field].append(rule)

    def add_type_validator(
        self,
        type_class: Type,
        validator: Callable[[Any], ValidationResult],
    ) -> None:
        """Add a validator for a specific type.

        Args:
            type_class: Type to validate.
            validator: Validation function.
        """
        self._type_validators[type_class] = validator

    def validate(
        self,
        data: dict[str, Any],
        schema: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Validate data against rules and schema.

        Args:
            data: Data to validate.
            schema: Optional schema definition.

        Returns:
            Validation result.
        """
        result = ValidationResult(is_valid=True)

        # Apply field rules
        for field, rules in self._rules.items():
            if field in data:
                value = data[field]
                for rule in rules:
                    is_valid, message = rule(value)
                    if not is_valid:
                        result.add_error(field, message, value)

        # Apply schema validation
        if schema:
            schema_result = self._validate_schema(data, schema)
            result.merge(schema_result)

        # Apply type validators
        for field, value in data.items():
            value_type = type(value)
            if value_type in self._type_validators:
                type_result = self._type_validators[value_type](value)
                result.merge(type_result)

        return result

    def validate_required(
        self,
        data: dict[str, Any],
        required_fields: list[str],
    ) -> ValidationResult:
        """Validate that required fields are present.

        Args:
            data: Data to validate.
            required_fields: List of required field names.

        Returns:
            Validation result.
        """
        result = ValidationResult(is_valid=True)

        for field in required_fields:
            if field not in data or data[field] is None:
                result.add_error(field, f"Required field '{field}' is missing")

        return result

    def validate_type(
        self,
        value: Any,
        expected_type: Type,
        field: str = "value",
    ) -> ValidationResult:
        """Validate value type.

        Args:
            value: Value to validate.
            expected_type: Expected type.
            field: Field name for error message.

        Returns:
            Validation result.
        """
        result = ValidationResult(is_valid=True)

        if not isinstance(value, expected_type):
            result.add_error(
                field,
                f"Expected type {expected_type.__name__}, got {type(value).__name__}",
                value,
            )

        return result

    def validate_range(
        self,
        value: float | int,
        min_val: float | int | None = None,
        max_val: float | int | None = None,
        field: str = "value",
    ) -> ValidationResult:
        """Validate value is within range.

        Args:
            value: Value to validate.
            min_val: Minimum value (inclusive).
            max_val: Maximum value (inclusive).
            field: Field name for error message.

        Returns:
            Validation result.
        """
        result = ValidationResult(is_valid=True)

        if min_val is not None and value < min_val:
            result.add_error(field, f"Value {value} is below minimum {min_val}")

        if max_val is not None and value > max_val:
            result.add_error(field, f"Value {value} is above maximum {max_val}")

        return result

    def validate_pattern(
        self,
        value: str,
        pattern: str,
        field: str = "value",
    ) -> ValidationResult:
        """Validate string matches pattern.

        Args:
            value: Value to validate.
            pattern: Regex pattern.
            field: Field name for error message.

        Returns:
            Validation result.
        """
        result = ValidationResult(is_valid=True)

        if not re.match(pattern, value):
            result.add_error(
                field,
                f"Value '{value}' does not match pattern '{pattern}'",
            )

        return result

    def validate_length(
        self,
        value: str | list | dict,
        min_len: int | None = None,
        max_len: int | None = None,
        field: str = "value",
    ) -> ValidationResult:
        """Validate collection length.

        Args:
            value: Value to validate.
            min_len: Minimum length.
            max_len: Maximum length.
            field: Field name for error message.

        Returns:
            Validation result.
        """
        result = ValidationResult(is_valid=True)
        length = len(value)

        if min_len is not None and length < min_len:
            result.add_error(field, f"Length {length} is below minimum {min_len}")

        if max_len is not None and length > max_len:
            result.add_error(field, f"Length {length} is above maximum {max_len}")

        return result

    def validate_choices(
        self,
        value: Any,
        choices: list[Any],
        field: str = "value",
    ) -> ValidationResult:
        """Validate value is one of allowed choices.

        Args:
            value: Value to validate.
            choices: Allowed values.
            field: Field name for error message.

        Returns:
            Validation result.
        """
        result = ValidationResult(is_valid=True)

        if value not in choices:
            result.add_error(
                field,
                f"Value '{value}' is not one of: {choices}",
            )

        return result

    def _validate_schema(
        self,
        data: dict[str, Any],
        schema: dict[str, Any],
    ) -> ValidationResult:
        """Validate data against schema.

        Args:
            data: Data to validate.
            schema: Schema definition.

        Returns:
            Validation result.
        """
        result = ValidationResult(is_valid=True)

        for field, field_schema in schema.items():
            # Check required
            if field_schema.get("required", False) and field not in data:
                result.add_error(field, f"Required field '{field}' is missing")
                continue

            if field not in data:
                continue

            value = data[field]

            # Check type
            if "type" in field_schema:
                type_result = self.validate_type(
                    value,
                    field_schema["type"],
                    field,
                )
                result.merge(type_result)

            # Check range
            if "min" in field_schema or "max" in field_schema:
                range_result = self.validate_range(
                    value,
                    field_schema.get("min"),
                    field_schema.get("max"),
                    field,
                )
                result.merge(range_result)

            # Check pattern
            if "pattern" in field_schema:
                pattern_result = self.validate_pattern(
                    value,
                    field_schema["pattern"],
                    field,
                )
                result.merge(pattern_result)

            # Check choices
            if "choices" in field_schema:
                choices_result = self.validate_choices(
                    value,
                    field_schema["choices"],
                    field,
                )
                result.merge(choices_result)

        return result


# Common validation rules
def required(value: Any) -> tuple[bool, str]:
    """Validate value is not None or empty."""
    if value is None or value == "":
        return False, "Value is required"
    return True, ""


def not_empty(value: str | list | dict) -> tuple[bool, str]:
    """Validate collection is not empty."""
    if len(value) == 0:
        return False, "Value cannot be empty"
    return True, ""


def positive(value: float | int) -> tuple[bool, str]:
    """Validate value is positive."""
    if value <= 0:
        return False, "Value must be positive"
    return True, ""


def non_negative(value: float | int) -> tuple[bool, str]:
    """Validate value is non-negative."""
    if value < 0:
        return False, "Value cannot be negative"
    return True, ""


def email(value: str) -> tuple[bool, str]:
    """Validate email format."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(pattern, value):
        return False, "Invalid email format"
    return True, ""
