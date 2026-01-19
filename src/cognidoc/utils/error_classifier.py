"""
Error classification module for checkpoint/resume functionality.

Classifies exceptions from LLM providers to determine appropriate action:
- QUOTA_EXHAUSTED: Daily/monthly quota exceeded, requires user action
- RATE_LIMITED: Temporary rate limit, can retry with backoff
- TRANSIENT: Network/connection issues, can retry
- PERMANENT: Invalid request or other permanent error, skip item
"""

from enum import Enum
from typing import Tuple


class ErrorType(Enum):
    """Classification of LLM provider errors."""

    QUOTA_EXHAUSTED = "RESOURCE_EXHAUSTED"  # Quota épuisé, reprendre plus tard
    RATE_LIMITED = "RATE_LIMITED"  # Temporaire, retry avec backoff
    TRANSIENT = "TRANSIENT"  # Erreur réseau, retry possible
    PERMANENT = "PERMANENT"  # Erreur permanente, skip


# Patterns indicating quota exhaustion (daily/monthly limits)
QUOTA_PATTERNS = [
    "resource_exhausted",
    "quota",
    "exceeded your current quota",
    "billing",
    "insufficient_quota",
    "quota_exceeded",
]

# Patterns indicating rate limiting (temporary, can retry)
RATE_LIMIT_PATTERNS = [
    "rate limit",
    "ratelimit",
    "rate_limit",
    "429",
    "too many requests",
    "request rate exceeded",
    "throttl",
]

# Patterns indicating transient/network errors
TRANSIENT_PATTERNS = [
    "timeout",
    "connection",
    "network",
    "temporarily unavailable",
    "service unavailable",
    "503",
    "502",
    "504",
    "gateway",
]

# Exception type names that indicate quota/rate errors
QUOTA_EXCEPTION_TYPES = [
    "ResourceExhausted",
    "QuotaExceeded",
]

RATE_LIMIT_EXCEPTION_TYPES = [
    "RateLimitError",
    "TooManyRequestsError",
]

TRANSIENT_EXCEPTION_TYPES = [
    "ConnectionError",
    "TimeoutError",
    "HTTPStatusError",
    "ConnectError",
    "ReadTimeout",
    "WriteTimeout",
]


def classify_error(exception: Exception) -> ErrorType:
    """
    Classify an exception to determine the appropriate action.

    Args:
        exception: The exception to classify

    Returns:
        ErrorType indicating the category of error
    """
    error_str = str(exception).lower()
    exc_type = type(exception).__name__

    # Check for quota exhaustion first (most important for checkpoint)
    if any(pattern in error_str for pattern in QUOTA_PATTERNS):
        return ErrorType.QUOTA_EXHAUSTED

    if exc_type in QUOTA_EXCEPTION_TYPES:
        return ErrorType.QUOTA_EXHAUSTED

    # Check for rate limits (can retry with backoff)
    if any(pattern in error_str for pattern in RATE_LIMIT_PATTERNS):
        return ErrorType.RATE_LIMITED

    if exc_type in RATE_LIMIT_EXCEPTION_TYPES:
        return ErrorType.RATE_LIMITED

    # Check for transient errors (network issues)
    if any(pattern in error_str for pattern in TRANSIENT_PATTERNS):
        # But if it's a 429, it's rate limiting
        if "429" in error_str:
            return ErrorType.RATE_LIMITED
        return ErrorType.TRANSIENT

    if exc_type in TRANSIENT_EXCEPTION_TYPES:
        if "429" in error_str:
            return ErrorType.RATE_LIMITED
        return ErrorType.TRANSIENT

    # Default to permanent error (bad request, invalid input, etc.)
    return ErrorType.PERMANENT


def is_quota_or_rate_error(exception: Exception) -> bool:
    """
    Check if an exception is a quota or rate limit error.

    These errors indicate the pipeline should stop and checkpoint.

    Args:
        exception: The exception to check

    Returns:
        True if quota or rate limit error, False otherwise
    """
    error_type = classify_error(exception)
    return error_type in (ErrorType.QUOTA_EXHAUSTED, ErrorType.RATE_LIMITED)


def is_retriable_error(exception: Exception) -> bool:
    """
    Check if an exception is potentially retriable.

    Args:
        exception: The exception to check

    Returns:
        True if the error might succeed on retry
    """
    error_type = classify_error(exception)
    return error_type in (ErrorType.RATE_LIMITED, ErrorType.TRANSIENT)


def get_error_info(exception: Exception) -> Tuple[ErrorType, str]:
    """
    Get error type and a clean error message.

    Args:
        exception: The exception to analyze

    Returns:
        Tuple of (ErrorType, clean_message)
    """
    error_type = classify_error(exception)
    # Truncate long error messages
    message = str(exception)[:500]
    return error_type, message
