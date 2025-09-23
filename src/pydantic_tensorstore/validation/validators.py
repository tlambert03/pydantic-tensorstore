"""High-level validation functions for TensorStore specifications."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from pydantic_tensorstore.core.spec import SpecValidator, TensorStoreSpec
from pydantic_tensorstore.validation.errors import (
    DriverValidationError,
    TensorStoreValidationError,
)


class TensorStoreValidator:
    """High-level validator for TensorStore specifications.

    Provides comprehensive validation with enhanced error reporting
    and optional TensorStore integration for runtime validation.
    """

    @staticmethod
    def validate_spec_dict(
        spec_dict: dict[str, Any],
        strict: bool = True,
        check_tensorstore: bool = False,
    ) -> dict[str, Any]:
        """Validate a specification dictionary.

        Parameters
        ----------
        spec_dict : dict
            Raw specification dictionary to validate
        strict : bool, default True
            If True, performs strict validation including cross-field checks
        check_tensorstore : bool, default False
            If True, attempts to validate against actual TensorStore (requires tensorstore package)

        Returns
        -------
        dict
            Validated and potentially normalized specification dictionary

        Raises
        ------
        TensorStoreValidationError
            If validation fails
        """
        try:
            # Basic structure validation
            validated_dict = SpecValidator.validate_spec_dict(spec_dict)

            if strict:
                # Perform additional cross-field validation
                TensorStoreValidator._validate_cross_field_consistency(validated_dict)

            if check_tensorstore:
                # Validate against actual TensorStore library
                TensorStoreValidator._validate_with_tensorstore(validated_dict)

            return validated_dict

        except Exception as e:
            if isinstance(e, TensorStoreValidationError):
                raise
            else:
                # Wrap other exceptions in our custom error type
                raise TensorStoreValidationError(
                    f"Validation failed: {e}",
                    spec_data=spec_dict,
                ) from e

    @staticmethod
    def validate_spec(
        spec: dict[str, Any] | TensorStoreSpec,
        strict: bool = True,
        check_tensorstore: bool = False,
    ) -> TensorStoreSpec:
        """Validate and parse a TensorStore specification.

        Parameters
        ----------
        spec : dict or TensorStoreSpec
            Specification to validate
        strict : bool, default True
            If True, performs strict validation
        check_tensorstore : bool, default False
            If True, validates against actual TensorStore

        Returns
        -------
        TensorStoreSpec
            Validated specification object

        Raises
        ------
        TensorStoreValidationError
            If validation fails
        """
        if isinstance(spec, dict):
            # Validate dictionary first
            validated_dict = TensorStoreValidator.validate_spec_dict(
                spec, strict=strict, check_tensorstore=check_tensorstore
            )

            # Parse into Pydantic model
            try:
                return TensorStoreSpec.model_validate(validated_dict)
            except ValidationError as e:
                raise TensorStoreValidationError(
                    f"Failed to parse specification: {e}",
                    spec_data=spec,
                ) from e

        elif isinstance(spec, TensorStoreSpec):
            # Already a parsed spec, validate if requested
            if strict or check_tensorstore:
                spec_dict = spec.model_dump()
                TensorStoreValidator.validate_spec_dict(
                    spec_dict, strict=strict, check_tensorstore=check_tensorstore
                )
            return spec

        else:
            raise TensorStoreValidationError(
                f"Invalid spec type: {type(spec)}. Must be dict or TensorStoreSpec."
            )

    @staticmethod
    def _validate_cross_field_consistency(spec_dict: dict[str, Any]) -> None:
        """Validate consistency between different fields."""
        driver = spec_dict.get("driver")
        if not driver:
            return  # Already validated in SpecValidator

        # Schema validation
        schema = spec_dict.get("schema")
        if schema:
            TensorStoreValidator._validate_schema_consistency(schema, driver)

        # Transform validation
        transform = spec_dict.get("transform")
        if transform and schema:
            TensorStoreValidator._validate_transform_schema_consistency(
                transform, schema
            )

        # Driver-specific cross-field validation
        if driver == "array":
            TensorStoreValidator._validate_array_consistency(spec_dict)
        elif driver == "zarr":
            TensorStoreValidator._validate_zarr_consistency(spec_dict)
        elif driver == "zarr3":
            TensorStoreValidator._validate_zarr3_consistency(spec_dict)
        elif driver == "n5":
            TensorStoreValidator._validate_n5_consistency(spec_dict)

    @staticmethod
    def _validate_schema_consistency(schema: dict[str, Any], driver: str) -> None:
        """Validate schema consistency."""
        # Check dtype compatibility with driver
        dtype = schema.get("dtype")
        if dtype and driver == "n5":
            # N5 has limited dtype support
            n5_types = {
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "int8",
                "int16",
                "int32",
                "int64",
                "float32",
                "float64",
            }
            if dtype not in n5_types:
                raise DriverValidationError(
                    f"N5 driver doesn't support dtype '{dtype}'",
                    driver=driver,
                    field_path="schema.dtype",
                )

        # Validate domain/rank consistency
        domain = schema.get("domain")
        rank = schema.get("rank")
        if domain and rank:
            domain_shape = domain.get("shape")
            if domain_shape and len(domain_shape) != rank:
                raise TensorStoreValidationError(
                    f"Domain shape rank {len(domain_shape)} doesn't match schema rank {rank}",
                    field_path="schema",
                )

    @staticmethod
    def _validate_transform_schema_consistency(
        transform: dict[str, Any], schema: dict[str, Any]
    ) -> None:
        """Validate transform and schema consistency."""
        # Check rank compatibility
        input_rank = transform.get("input_rank")
        schema_rank = schema.get("rank")

        if input_rank is not None and schema_rank is not None:
            if input_rank != schema_rank:
                raise TensorStoreValidationError(
                    f"Transform input rank {input_rank} doesn't match schema rank {schema_rank}",
                    field_path="transform",
                )

    @staticmethod
    def _validate_array_consistency(spec_dict: dict[str, Any]) -> None:
        """Validate array driver consistency."""
        array_data = spec_dict.get("array")
        dtype = spec_dict.get("dtype")

        if not array_data or not dtype:
            return  # Missing required fields already caught

        # Basic shape consistency would be validated here
        # For now, just ensure both are present
        pass

    @staticmethod
    def _validate_zarr_consistency(spec_dict: dict[str, Any]) -> None:
        """Validate Zarr driver consistency."""
        kvstore = spec_dict.get("kvstore")
        if not kvstore:
            raise DriverValidationError(
                "Zarr driver requires kvstore specification",
                driver="zarr",
                field_path="kvstore",
            )

    @staticmethod
    def _validate_zarr3_consistency(spec_dict: dict[str, Any]) -> None:
        """Validate Zarr3 driver consistency."""
        kvstore = spec_dict.get("kvstore")
        if not kvstore:
            raise DriverValidationError(
                "Zarr3 driver requires kvstore specification",
                driver="zarr3",
                field_path="kvstore",
            )

    @staticmethod
    def _validate_n5_consistency(spec_dict: dict[str, Any]) -> None:
        """Validate N5 driver consistency."""
        kvstore = spec_dict.get("kvstore")
        if not kvstore:
            raise DriverValidationError(
                "N5 driver requires kvstore specification",
                driver="n5",
                field_path="kvstore",
            )

    @staticmethod
    def _validate_with_tensorstore(spec_dict: dict[str, Any]) -> None:
        """Validate against actual TensorStore library if available."""
        try:
            import tensorstore as ts  # type: ignore

            # Try to create a spec with TensorStore
            try:
                ts.Spec(spec_dict)
            except Exception as e:
                raise TensorStoreValidationError(
                    f"TensorStore validation failed: {e}",
                    spec_data=spec_dict,
                ) from e

        except ImportError:
            # TensorStore not available, skip this validation
            pass


# Convenience functions
def validate_spec(
    spec: dict[str, Any] | TensorStoreSpec,
    strict: bool = True,
    check_tensorstore: bool = False,
) -> TensorStoreSpec:
    """Validate a TensorStore specification.

    Parameters
    ----------
    spec : dict or TensorStoreSpec
        Specification to validate
    strict : bool, default True
        If True, performs strict validation
    check_tensorstore : bool, default False
        If True, validates against actual TensorStore

    Returns
    -------
    TensorStoreSpec
        Validated specification object
    """
    return TensorStoreValidator.validate_spec(
        spec, strict=strict, check_tensorstore=check_tensorstore
    )


def validate_spec_dict(
    spec_dict: dict[str, Any],
    strict: bool = True,
    check_tensorstore: bool = False,
) -> dict[str, Any]:
    """Validate a specification dictionary.

    Parameters
    ----------
    spec_dict : dict
        Raw specification dictionary to validate
    strict : bool, default True
        If True, performs strict validation
    check_tensorstore : bool, default False
        If True, validates against actual TensorStore

    Returns
    -------
    dict
        Validated and potentially normalized specification dictionary
    """
    return TensorStoreValidator.validate_spec_dict(
        spec_dict, strict=strict, check_tensorstore=check_tensorstore
    )
