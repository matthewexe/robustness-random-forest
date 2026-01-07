from typing import TypeVar

from robustness.domain.logging import get_logger

T = TypeVar("T")

logger = get_logger(__name__)

def from_json(json_content: str, cls: T) -> T:
    from pydantic import TypeAdapter

    logger.info(f"Deserializing JSON to {cls}")
    logger.debug(f"JSON content length: {len(json_content)}")

    adapter = TypeAdapter(cls)
    result = adapter.validate_json(json_content)
    logger.info(f"Successfully deserialized JSON to {cls}")
    return result

