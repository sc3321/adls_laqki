
class QPEError(Exception):
    """Base exception for all QPE errors"""
    pass

class InfeasibleError(QPEError):
    """
    Raised when solver cannot find any valid precision assignment that satisfies all constraints simultaneously
    """
    
    def __init__(self, message: str, diagnostics: dict | None = None):
        self.diagnostics = diagnostics or {}
        super().__init__(message)

class SchemaVersionError(QPEError):
    """
    Raised when cached data was written with a different schema version than current QPE release expects
    
    Prevents silent data corruption from loading stale cache entries whose fields have 
    changed semantics or been renamed/removed
    """
    
    def __init__(self, actual_version: int, expected_version: int, cache_path: str):
        self.actual_version = actual_version
        self.expected_version = expected_version
        self.cache_path = cache_path
        super().__init__(
            f"Cached data at {cache_path} has schema version {actual_version}, "
            f"but current QPE expects version {expected_version} - "
            f"delete the cache directory and re-run to regenerate"
        )