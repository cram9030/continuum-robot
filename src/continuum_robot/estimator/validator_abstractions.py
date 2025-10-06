"""
Abstract interfaces and data structures for estimator validation.

This module implements the Strategy Pattern for validation, allowing different
validation strategies to be applied to estimators without coupling validation
logic to the estimator implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels for validation messages."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """
    Result of validation with support for multiple messages.

    Attributes:
        is_valid: Overall validation status (False if any errors present)
        messages: List of (severity, message) tuples
    """

    is_valid: bool = True
    messages: List[Tuple[ValidationSeverity, str]] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add an error message and mark validation as failed."""
        self.messages.append((ValidationSeverity.ERROR, message))
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message (does not fail validation)."""
        self.messages.append((ValidationSeverity.WARNING, message))

    def add_info(self, message: str) -> None:
        """Add an informational message."""
        self.messages.append((ValidationSeverity.INFO, message))

    def get_errors(self) -> List[str]:
        """Get all error messages."""
        return [msg for sev, msg in self.messages if sev == ValidationSeverity.ERROR]

    def get_warnings(self) -> List[str]:
        """Get all warning messages."""
        return [msg for sev, msg in self.messages if sev == ValidationSeverity.WARNING]

    def get_info(self) -> List[str]:
        """Get all informational messages."""
        return [msg for sev, msg in self.messages if sev == ValidationSeverity.INFO]

    def has_errors(self) -> bool:
        """Check if any errors are present."""
        return not self.is_valid

    def has_warnings(self) -> bool:
        """Check if any warnings are present."""
        return any(sev == ValidationSeverity.WARNING for sev, _ in self.messages)

    def format_messages(self, include_info: bool = False) -> str:
        """
        Format all messages as a human-readable string.

        Args:
            include_info: Whether to include informational messages

        Returns:
            Formatted string with all messages
        """
        lines = []
        for severity, message in self.messages:
            if not include_info and severity == ValidationSeverity.INFO:
                continue
            lines.append(f"[{severity.value.upper()}] {message}")
        return "\n".join(lines)


class IEstimatorValidator(ABC):
    """
    Abstract base class for estimator validation strategies.

    Implementations of this interface define specific validation logic for
    different types of estimators (linear, nonlinear, etc.).
    """

    @abstractmethod
    def validate_initialization(self, **kwargs) -> ValidationResult:
        """
        Validate estimator initialization parameters.

        This method should check all initialization parameters for:
        - Correct types and dimensions
        - Valid numerical properties (no NaN/inf, proper matrix conditions)
        - System-specific properties (observability, stability, etc.)

        Args:
            **kwargs: Initialization parameters specific to the estimator type

        Returns:
            ValidationResult with validation status and messages
        """
        pass

    @abstractmethod
    def validate_runtime_state(self, **kwargs) -> ValidationResult:
        """
        Validate runtime state during estimation.

        This method should perform lightweight checks during estimation:
        - State vector dimensions and validity
        - Measurement vector dimensions and validity
        - Input vector dimensions and validity

        Args:
            **kwargs: Runtime parameters specific to the estimator type

        Returns:
            ValidationResult with validation status and messages
        """
        pass
