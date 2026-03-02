
class QPEError(Exception):
    """Base exception for all QPE errors"""
    pass

class InfeasibleError(QPEError):
    """
    Raised when solver cannot find any valid precision assignment
    that satisfies all constraints simultaneously.
    """

    def __init__(self, message: str, diagnostics: dict | None = None):
        self.diagnostics = diagnostics or {}
        super().__init__(message)


class SchemaVersionError(QPEError):
    """
    Raised when cached data was written with a different schema version
    than the current QPE release expects
    """

    def __init__(self, message: str):
        super().__init__(message)
