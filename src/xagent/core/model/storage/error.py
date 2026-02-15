class StorageError(Exception):
    """Base class for storage-related errors."""

    pass


class StorageWriteError(StorageError):
    """Raised when writing to storage fails."""

    path: str
    reason: str

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        self.message = f"Failed to write to {path}: {reason}"


class StorageReadError(StorageError):
    """Raised when reading from storage fails."""

    path: str
    reason: str

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        self.message = f"Failed to read from {path}: {reason}"


class InvalidModelError(StorageError):
    """Raised when parsing model from storage fails."""

    path: str
    reason: str

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        self.message = f"Failed to parse model from {path}: {reason}"
