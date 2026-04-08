"""
Secure logging with automatic masking of sensitive fields.
Prevents tokens, passwords, email bodies from appearing in logs.
"""
import re
import logging
from logging.handlers import RotatingFileHandler
import os


# Patterns to mask in log output
_MASK_PATTERNS = [
    # OAuth tokens (long alphanumeric strings)
    (re.compile(r'(access_token["\s:=]+)[A-Za-z0-9_\-\.]{20,}', re.IGNORECASE),
     r'\1[REDACTED]'),
    (re.compile(r'(refresh_token["\s:=]+)[A-Za-z0-9_\-\.]{20,}', re.IGNORECASE),
     r'\1[REDACTED]'),
    # Bearer tokens
    (re.compile(r'(Bearer\s+)[A-Za-z0-9_\-\.]{20,}', re.IGNORECASE),
     r'\1[REDACTED]'),
    # API keys
    (re.compile(r'(api_key["\s:=]+)[A-Za-z0-9_\-]{20,}', re.IGNORECASE),
     r'\1[REDACTED]'),
    (re.compile(r'gsk_[A-Za-z0-9]{40,}'),
     '[GROQ_KEY_REDACTED]'),
    # Email bodies (truncate long body strings in logs)
    (re.compile(r'(body["\s:=]+")[^"]{200,}"', re.IGNORECASE),
     r'\1[EMAIL_BODY_TRUNCATED]"'),
    # Encryption keys
    (re.compile(r'(key["\s:=]+)[A-Za-z0-9+/=]{32,}', re.IGNORECASE),
     r'\1[KEY_REDACTED]'),
    # Client secrets
    (re.compile(r'(client_secret["\s:=]+)[A-Za-z0-9_\-]{10,}', re.IGNORECASE),
     r'\1[SECRET_REDACTED]'),
]


class SensitiveDataFilter(logging.Filter):
    """Logging filter that masks sensitive data before output."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Mask the message
        msg = record.getMessage()
        for pattern, replacement in _MASK_PATTERNS:
            msg = pattern.sub(replacement, msg)
        record.msg  = msg
        record.args = ()   # clear args since we've already formatted
        return True


def get_secure_logger(name: str) -> logging.Logger:
    """
    Get a logger with sensitive data masking applied.
    Replaces the standard setup_logger for security-critical modules.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        level   = os.getenv("LOG_LEVEL","INFO").upper()
        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, level, logging.INFO))
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        handler.addFilter(SensitiveDataFilter())
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level, logging.INFO))

        # Also add file handler if logs/ exists
        os.makedirs("logs", exist_ok=True)
        fh = RotatingFileHandler(
            "logs/aeoa.log", maxBytes=5*1024*1024, backupCount=3
        )
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))
        fh.addFilter(SensitiveDataFilter())
        logger.addHandler(fh)

    return logger