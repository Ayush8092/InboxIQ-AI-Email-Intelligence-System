import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(
    os.path.dirname(os.path.dirname(__file__)), ".env"
))

APP_ENV              = os.getenv("APP_ENV", "development")
GROQ_API_KEY         = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL           = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_RETRIES          = int(os.getenv("MAX_RETRIES", "2"))

# On Render disk is mounted at /data
# Locally use data/aeoa.db
_default_db = "/data/aeoa.db" if APP_ENV == "production" else "data/aeoa.db"
DB_PATH     = os.getenv("DB_PATH", _default_db)

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
OAUTH_REDIRECT_URI   = os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8501")

AEOA_ENCRYPTION_KEY  = os.getenv("AEOA_ENCRYPTION_KEY", "")
JWT_SECRET           = os.getenv("JWT_SECRET", "")
AEOA_API_KEY         = os.getenv("AEOA_API_KEY", "")
AEOA_ADMIN_PASSWORD  = os.getenv("AEOA_ADMIN_PASSWORD", "admin123")

SENDER_IMPORTANCE = {
    "boss@company.com":   10,
    "cto@company.com":    10,
    "hr@company.com":      7,
    "client@bigcorp.com":  9,
}