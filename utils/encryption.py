"""
Encryption utilities with strict key management.
Production: AEOA_ENCRYPTION_KEY must be set as env var.
Development: auto-generates key but warns clearly.
Fails fast if key is invalid — never silently uses wrong key.
"""
import os
import base64
from cryptography.fernet import Fernet, InvalidToken
from utils.secure_logger import get_secure_logger

logger = get_secure_logger(__name__)

_KEY: bytes | None = None
_KEY_SOURCE: str   = "none"


def _get_key() -> bytes:
    """
    Get encryption key with strict validation.
    Priority:
    1. AEOA_ENCRYPTION_KEY env var (REQUIRED in production)
    2. Auto-generate for development (warns loudly)

    Fails fast if env key is malformed.
    """
    global _KEY, _KEY_SOURCE

    if _KEY is not None:
        return _KEY

    env_key = os.getenv("AEOA_ENCRYPTION_KEY", "").strip()
    app_env = os.getenv("APP_ENV", "development").lower()

    if env_key:
        try:
            key_bytes = env_key.encode() if isinstance(env_key, str) else env_key
            Fernet(key_bytes)   # validate format
            _KEY        = key_bytes
            _KEY_SOURCE = "environment"
            logger.info("Encryption key loaded from AEOA_ENCRYPTION_KEY")
            return _KEY
        except Exception as e:
            msg = f"AEOA_ENCRYPTION_KEY is set but invalid: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    # Production must have the key set
    if app_env == "production":
        msg = (
            "CRITICAL: AEOA_ENCRYPTION_KEY is not set. "
            "This is REQUIRED in production to prevent token loss on restart. "
            "Generate a key with: python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\" "
            "then set it as AEOA_ENCRYPTION_KEY environment variable."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    # Development only — generate fresh key with clear warning
    _KEY        = Fernet.generate_key()
    _KEY_SOURCE = "generated"
    logger.warning(
        "WARNING: No AEOA_ENCRYPTION_KEY set. "
        "Generated fresh key — tokens will be undecryptable after restart. "
        "Set AEOA_ENCRYPTION_KEY for persistent sessions."
    )
    return _KEY


def get_key_source() -> str:
    """Return where the current encryption key came from."""
    return _KEY_SOURCE


def generate_new_key() -> str:
    """
    Generate a new Fernet key for use as AEOA_ENCRYPTION_KEY.
    Run once: python -c "from utils.encryption import generate_new_key; print(generate_new_key())"
    """
    return Fernet.generate_key().decode("utf-8")


def encrypt_token(token: str) -> str:
    """Encrypt token. Returns empty string if token is empty."""
    if not token:
        return ""
    try:
        f         = Fernet(_get_key())
        encrypted = f.encrypt(token.encode("utf-8"))
        return base64.urlsafe_b64encode(encrypted).decode("utf-8")
    except Exception as e:
        logger.error(f"Token encryption failed: {type(e).__name__}")
        return ""


def decrypt_token(encrypted_token: str) -> str:
    """
    Decrypt token. Returns empty string on failure.
    InvalidToken means key mismatch (restart without persistent key).
    """
    if not encrypted_token:
        return ""
    try:
        f         = Fernet(_get_key())
        raw       = base64.urlsafe_b64decode(encrypted_token.encode("utf-8"))
        decrypted = f.decrypt(raw)
        return decrypted.decode("utf-8")
    except InvalidToken:
        logger.warning(
            "Token decryption failed — key mismatch. "
            "User must re-authenticate (expected after restart without persistent key)."
        )
        return ""
    except Exception as e:
        logger.error(f"Token decryption error: {type(e).__name__}")
        return ""