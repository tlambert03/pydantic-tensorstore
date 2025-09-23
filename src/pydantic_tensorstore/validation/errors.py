"""Custom exception classes for TensorStore validation errors."""

from __future__ import annotations

from typing import Any


class TensorStoreValidationError(ValueError):
    """Base class for TensorStore validation errors.

    Provides enhanced error messages with field paths and context.
    """

    def __init__(
        self,
        message: str,
        field_path: str | None = None,
        spec_data: dict[str, Any] | None = None,
        driver: str | None = None,
    ) -> None:
        """Initialize validation error.

        Parameters
        ----------
        message : str
            Error message describing the validation failure
        field_path : str, optional
            Path to the field that failed validation (e.g., "schema.dtype")
        spec_data : dict, optional
            The spec data that failed validation
        driver : str, optional
            The driver being validated
        """
        self.field_path = field_path
        self.spec_data = spec_data
        self.driver = driver

        # Build enhanced error message
        parts = []
        if driver:
            parts.append(f"Driver '{driver}':")
        if field_path:
            parts.append(f"Field '{field_path}':")
        parts.append(message)

        enhanced_message = " ".join(parts)
        super().__init__(enhanced_message)

    def __str__(self) -> str:
        """Return string representation of the error."""
        return super().__str__()


class DriverValidationError(TensorStoreValidationError):
    """Error in driver-specific validation."""

    def __init__(
        self,
        message: str,
        driver: str,
        field_path: str | None = None,
        spec_data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize driver validation error."""
        super().__init__(
            message=message,
            field_path=field_path,
            spec_data=spec_data,
            driver=driver,
        )


class SchemaValidationError(TensorStoreValidationError):
    """Error in schema validation."""

    def __init__(
        self,
        message: str,
        field_path: str | None = None,
        schema_data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize schema validation error."""
        super().__init__(
            message=message,
            field_path=field_path,
            spec_data=schema_data,
        )


class KvStoreValidationError(TensorStoreValidationError):
    """Error in key-value store validation."""

    def __init__(
        self,
        message: str,
        kvstore_driver: str | None = None,
        field_path: str | None = None,
        kvstore_data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize kvstore validation error."""
        super().__init__(
            message=message,
            field_path=field_path,
            spec_data=kvstore_data,
            driver=kvstore_driver,
        )


class TransformValidationError(TensorStoreValidationError):
    """Error in index transform validation."""

    def __init__(
        self,
        message: str,
        field_path: str | None = None,
        transform_data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize transform validation error."""
        super().__init__(
            message=message,
            field_path=field_path,
            spec_data=transform_data,
        )


class ContextValidationError(TensorStoreValidationError):
    """Error in context validation."""

    def __init__(
        self,
        message: str,
        resource_name: str | None = None,
        field_path: str | None = None,
        context_data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize context validation error."""
        enhanced_message = message
        if resource_name:
            enhanced_message = f"Context resource '{resource_name}': {message}"

        super().__init__(
            message=enhanced_message,
            field_path=field_path,
            spec_data=context_data,
        )
