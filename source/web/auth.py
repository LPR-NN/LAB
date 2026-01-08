"""Basic authentication for the web interface."""

import logging
import secrets
import time
from collections import defaultdict
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from source.settings import get_settings

logger = logging.getLogger(__name__)
security = HTTPBasic()

# Rate limiting: track failed attempts per IP
_failed_attempts: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT_WINDOW = 300  # 5 minutes
_MAX_FAILED_ATTEMPTS = 5  # Max failures before lockout


def _cleanup_old_attempts(ip: str) -> None:
    """Remove attempts older than the rate limit window."""
    now = time.time()
    _failed_attempts[ip] = [
        t for t in _failed_attempts[ip] if now - t < _RATE_LIMIT_WINDOW
    ]


def _is_rate_limited(ip: str) -> bool:
    """Check if IP is rate limited."""
    _cleanup_old_attempts(ip)
    return len(_failed_attempts[ip]) >= _MAX_FAILED_ATTEMPTS


def _record_failed_attempt(ip: str) -> None:
    """Record a failed login attempt."""
    _failed_attempts[ip].append(time.time())


def verify_credentials(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
    request: Request,
) -> str:
    """Verify basic auth credentials with rate limiting."""
    client_ip = request.client.host if request.client else "unknown"

    # Check rate limit before verifying credentials
    if _is_rate_limited(client_ip):
        logger.warning(
            "Rate limited: ip=%s path=%s",
            client_ip,
            request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed attempts. Try again later.",
            headers={"Retry-After": str(_RATE_LIMIT_WINDOW)},
        )

    settings = get_settings()

    # Check against all configured users
    is_valid = False
    for username, password in settings.auth_users.items():
        is_username_correct = secrets.compare_digest(
            credentials.username.encode("utf-8"),
            username.encode("utf-8"),
        )
        is_password_correct = secrets.compare_digest(
            credentials.password.encode("utf-8"),
            password.encode("utf-8"),
        )
        if is_username_correct and is_password_correct:
            is_valid = True
            break

    if not is_valid:
        _record_failed_attempt(client_ip)
        logger.warning(
            "Failed login attempt: user=%r ip=%s path=%s attempts=%d",
            credentials.username,
            client_ip,
            request.url.path,
            len(_failed_attempts[client_ip]),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    # Clear failed attempts on successful login
    if client_ip in _failed_attempts:
        del _failed_attempts[client_ip]

    return credentials.username
