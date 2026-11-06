"""One error type for everything the API refuses to do. app.py turns it into an HTTP status."""
from __future__ import annotations


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status, self.message = status, message
