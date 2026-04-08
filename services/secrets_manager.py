"""
Secrets management with multiple backends.
Priority:
1. AWS Secrets Manager (production)
2. HashiCorp Vault (production alternative)
3. Environment variables (development)

Never falls back to hardcoded defaults for sensitive values.
"""
import os
import json
from functools import lru_cache
from utils.secure_logger import get_secure_logger

logger = get_secure_logger(__name__)

_SECRET_BACKEND = os.getenv("SECRET_BACKEND", "env")   # env | aws | vault


@lru_cache(maxsize=50)
def _get_secret_aws(secret_name: str) -> dict | None:
    """Fetch secret from AWS Secrets Manager."""
    try:
        import boto3
        from botocore.exceptions import ClientError

        region = os.getenv("AWS_REGION", "us-east-1")
        client = boto3.client("secretsmanager", region_name=region)
        resp   = client.get_secret_value(SecretId=secret_name)
        raw    = resp.get("SecretString","")
        try:
            return json.loads(raw)
        except Exception:
            return {"value": raw}
    except ImportError:
        logger.warning("boto3 not installed — cannot use AWS Secrets Manager")
        return None
    except Exception as e:
        logger.error(f"AWS secret fetch failed: {type(e).__name__} | {secret_name}")
        return None


def _get_secret_vault(secret_path: str) -> dict | None:
    """Fetch secret from HashiCorp Vault."""
    try:
        import hvac
        vault_url   = os.getenv("VAULT_ADDR", "http://localhost:8200")
        vault_token = os.getenv("VAULT_TOKEN", "")
        if not vault_token:
            logger.warning("VAULT_TOKEN not set")
            return None
        client = hvac.Client(url=vault_url, token=vault_token)
        if not client.is_authenticated():
            logger.warning("Vault authentication failed")
            return None
        resp = client.secrets.kv.v2.read_secret_version(path=secret_path)
        return resp.get("data",{}).get("data",{})
    except ImportError:
        logger.warning("hvac not installed — cannot use Vault")
        return None
    except Exception as e:
        logger.error(f"Vault secret fetch failed: {type(e).__name__} | {secret_path}")
        return None


def get_secret(key: str, default: str | None = None) -> str:
    """
    Get a secret value from the configured backend.
    Falls back through: backend → env var → default.

    For production secrets (JWT_SECRET, API keys), raises if not found.
    """
    backend = _SECRET_BACKEND.lower()

    if backend == "aws":
        secret_name = os.getenv("AWS_SECRET_NAME", "aeoa/production")
        secrets     = _get_secret_aws(secret_name)
        if secrets and key in secrets:
            return secrets[key]

    elif backend == "vault":
        secret_path = os.getenv("VAULT_SECRET_PATH", "aeoa/production")
        secrets     = _get_secret_vault(secret_path)
        if secrets and key in secrets:
            return secrets[key]

    # Fallback: environment variable
    val = os.getenv(key, default)

    # Strict: never allow empty production secrets
    critical_keys = {
        "JWT_SECRET", "AEOA_ENCRYPTION_KEY", "GROQ_API_KEY",
        "GOOGLE_CLIENT_SECRET", "AEOA_API_KEY",
    }
    app_env = os.getenv("APP_ENV","development")
    if key in critical_keys and app_env == "production" and not val:
        raise RuntimeError(
            f"CRITICAL: Secret '{key}' is required in production but not set. "
            f"Configure SECRET_BACKEND=aws or SECRET_BACKEND=vault, "
            f"or set the environment variable."
        )

    return val or ""


def get_gmail_credentials() -> dict:
    """Get Gmail OAuth credentials from secrets backend."""
    return {
        "client_id":     get_secret("GOOGLE_CLIENT_ID"),
        "client_secret": get_secret("GOOGLE_CLIENT_SECRET"),
        "redirect_uri":  get_secret("OAUTH_REDIRECT_URI", "http://localhost:8501/oauth_callback"),
    }


def get_groq_api_key() -> str:
    return get_secret("GROQ_API_KEY")


def get_jwt_secret() -> str:
    return get_secret("JWT_SECRET", os.urandom(32).hex())


def get_encryption_key() -> str:
    return get_secret("AEOA_ENCRYPTION_KEY", "")