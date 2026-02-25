from error_handling.types import SchemaVersionError


CURRENT_SCHEMA_VERSION = 2


def validate_schema(
    data: dict,
    expected_version: int = CURRENT_SCHEMA_VERSION,
    source_path: str = "<unknown>",
) -> None:
    """
    Validate that serialized data matches current schema version
    
    Called when loading any cached data (scorer profiles, profiler results, FP16 baselines)
    Raises SchemaVersionError on mismatch, forcing cache regeneration rather than 
    silently loading incompatible data
    
    Args:
        data: Deserialized JSON/dict with a schema_version field
        expected_version: Version the current code expects
        source_path: File path for error messages
    
    Raises:
        SchemaVersionError: If versions do not match
    """
    actual = data.get("schema_version", 1)
    if actual != expected_version:
        raise SchemaVersionError(
            actual_version=actual,
            expected_version=expected_version,
            cache_path=source_path,
        )