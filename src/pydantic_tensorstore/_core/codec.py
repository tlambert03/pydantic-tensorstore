"""Just a stub for codec base class.

See pydantic_tensorstore.Codec for the full union type.
"""

from __future__ import annotations

from pydantic_tensorstore._core.base import TensorStoreModel


class CodecBase(TensorStoreModel):
    """Codecs are specified by a required driver property that identifies the driver.

    All other properties are driver-specific. Refer to the driver documentation for the
    supported codec drivers and the driver-specific properties.
    """

    # driver: str
