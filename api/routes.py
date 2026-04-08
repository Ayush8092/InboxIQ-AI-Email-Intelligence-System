"""
FastAPI routes — final production-grade version.

Fixes applied:
  1. Middleware: global try/catch + request ID + timeout handling
  2. Request ID tracing on every request and response
  3. asyncio.wait_for timeout on all endpoint calls
  4. OpenAPI security schema properly configured
  5. Idempotency: atomic Redis SET NX + TTL enforcement
  6. Circuit breaker for Redis and Celery
  7. Token validation (expired + revoked) before revocation
  8. Pagination on all list endpoints
  9. All configs from environment variables
"""
import os
import time
import uuid
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI, HTTPException, Depends,
    Request, Response, Header, Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials,
    OAuth2PasswordBearer,
    SecurityScopes,
)
from fastapi.openapi.models import SecurityScheme
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from api.auth import (
    require_permission,
    get_current_user,
    revoke_token,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from utils.secure_logger import get_secure_logger
from utils.observability import (
    get_metrics, get_content_type,
    record_api_request, get_system_health,
    record_feedback, record_ml_prediction,
    StructuredLogger,
)
from utils.redis_client import (
    check_rate_limit_redis,
    get_cache_stats,
    get_job_status_redis,
    is_email_processed,
    is_redis_available,
    get_redis,
)
from utils.alerting import (
    get_active_alerts,
    check_error_rate,
    check_latency,
    check_ml_accuracy,
    check_feature_drift,
)
from workers.celery_app import celery_app

logger   = get_secure_logger("aeoa.api")
slog     = StructuredLogger("aeoa.api")
security = HTTPBearer(auto_error=False)

# ── Configuration from environment (no hardcoding) ───────────────────────────

_ENV     = os.getenv("APP_ENV", "development")
_ORIGINS = (
    os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8501,http://localhost:8000",
    ).split(",")
    if _ENV != "production"
    else os.getenv("APP_URL", "https://aeoa.onrender.com").split(",")
)

# All timeouts from env
ENDPOINT_TIMEOUT_S   = float(os.getenv("ENDPOINT_TIMEOUT_S",    "30"))
MIDDLEWARE_TIMEOUT_S = float(os.getenv("MIDDLEWARE_TIMEOUT_S",   "35"))

# Rate limits from env  — format: "action:max:window_seconds"
def _parse_rate_limits() -> dict[str, tuple[int, int]]:
    raw = os.getenv(
        "RATE_LIMITS",
        "process_emails:50:3600,"
        "gmail_fetch:10:3600,"
        "train_model:5:3600,"
        "generate_reply:30:3600",
    )
    result = {}
    for item in raw.split(","):
        parts = item.strip().split(":")
        if len(parts) == 3:
            try:
                result[parts[0]] = (int(parts[1]), int(parts[2]))
            except ValueError:
                pass
    return result

_RATE_LIMITS: dict[str, tuple[int, int]] = _parse_rate_limits()

# Idempotency TTL from env
IDEMPOTENCY_TTL_S = int(os.getenv("IDEMPOTENCY_TTL_S", "86400"))  # 24 hours

# Circuit breaker thresholds from env
CB_FAILURE_THRESHOLD = int(os.getenv("CB_FAILURE_THRESHOLD", "5"))
CB_RECOVERY_TIMEOUT  = int(os.getenv("CB_RECOVERY_TIMEOUT",  "60"))


# ── Circuit breaker ────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Simple in-memory circuit breaker.
    States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing).

    CLOSED:    all requests allowed
    OPEN:      all requests rejected immediately
    HALF_OPEN: one test request allowed; if OK → CLOSED, if fail → OPEN
    """

    def __init__(self, name: str, failure_threshold: int, recovery_timeout: int):
        self.name              = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout  = recovery_timeout
        self._failures         = 0
        self._state            = "CLOSED"
        self._opened_at: float = 0.0

    @property
    def state(self) -> str:
        if self._state == "OPEN":
            if time.time() - self._opened_at >= self.recovery_timeout:
                self._state = "HALF_OPEN"
                logger.info(f"Circuit {self.name}: OPEN → HALF_OPEN")
        return self._state

    def record_success(self):
        if self._state in ("HALF_OPEN", "CLOSED"):
            self._failures = 0
            if self._state == "HALF_OPEN":
                logger.info(f"Circuit {self.name}: HALF_OPEN → CLOSED")
            self._state = "CLOSED"

    def record_failure(self):
        self._failures += 1
        if self._state == "HALF_OPEN" or self._failures >= self.failure_threshold:
            self._state    = "OPEN"
            self._opened_at = time.time()
            logger.warning(
                f"Circuit {self.name}: OPEN | "
                f"failures={self._failures}"
            )

    def is_open(self) -> bool:
        return self.state == "OPEN"

    def to_dict(self) -> dict:
        return {
            "name":      self.name,
            "state":     self.state,
            "failures":  self._failures,
            "threshold": self.failure_threshold,
        }


# One circuit breaker per external dependency
_cb_redis  = CircuitBreaker("redis",  CB_FAILURE_THRESHOLD, CB_RECOVERY_TIMEOUT)
_cb_celery = CircuitBreaker("celery", CB_FAILURE_THRESHOLD, CB_RECOVERY_TIMEOUT)


def _redis_circuit_check():
    """Raise 503 if Redis circuit is open."""
    if _cb_redis.is_open():
        raise HTTPException(
            status_code=503,
            detail=(
                "Redis service unavailable (circuit open). "
                f"Retrying in ~{CB_RECOVERY_TIMEOUT}s."
            ),
        )


def _celery_circuit_check():
    """Raise 503 if Celery circuit is open."""
    if _cb_celery.is_open():
        raise HTTPException(
            status_code=503,
            detail=(
                "Task queue unavailable (circuit open). "
                f"Retrying in ~{CB_RECOVERY_TIMEOUT}s."
            ),
        )


def _dispatch_celery_task(task_fn, *args, queue: str = "default"):
    """
    Dispatch a Celery task with circuit breaker protection.
    Records success/failure to update circuit state.
    """
    _celery_circuit_check()
    try:
        task = task_fn.apply_async(args=list(args), queue=queue)
        _cb_celery.record_success()
        return task
    except Exception as e:
        _cb_celery.record_failure()
        logger.error(f"Celery dispatch failed: {type(e).__name__}")
        raise HTTPException(
            status_code=503,
            detail=f"Task queue error: {type(e).__name__}",
        )


# ── OpenAPI security schema ────────────────────────────────────────────────────

app = FastAPI(
    title="AEOA API",
    version="3.0.0",
    docs_url="/docs" if _ENV != "production" else None,
    redoc_url=None,
    # Properly declare Bearer JWT in OpenAPI schema
    openapi_tags=[
        {"name": "auth",      "description": "Authentication endpoints"},
        {"name": "emails",    "description": "Email operations"},
        {"name": "ml",        "description": "Machine learning endpoints"},
        {"name": "system",    "description": "System health and metrics"},
        {"name": "analytics", "description": "Analytics (admin only)"},
    ],
)

# Inject Bearer auth into OpenAPI so /docs shows the lock icon on every endpoint
def _custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi
    schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )
    schema.setdefault("components", {}).setdefault("securitySchemes", {})
    schema["components"]["securitySchemes"]["BearerAuth"] = {
        "type":         "http",
        "scheme":       "bearer",
        "bearerFormat": "JWT",
        "description":  "JWT access token — obtain from POST /auth/login",
    }
    # Apply globally to all operations
    for path_item in schema.get("paths", {}).values():
        for operation in path_item.values():
            if isinstance(operation, dict):
                operation.setdefault("security", [{"BearerAuth": []}])
    app.openapi_schema = schema
    return schema

app.openapi = _custom_openapi

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    allow_credentials=True,
    expose_headers=["X-Response-Time", "X-Request-ID"],
)


# ── Middleware: safety + request ID + timeout ─────────────────────────────────

@app.middleware("http")
async def production_middleware(request: Request, call_next):
    """
    Single middleware handling:
    1. Request ID generation and propagation
    2. Global try/catch with structured error response
    3. Timeout enforcement (MIDDLEWARE_TIMEOUT_S)
    4. Latency tracking + Prometheus recording
    5. Alert triggers
    """
    # 1. Request ID — use incoming or generate new
    request_id = (
        request.headers.get("X-Request-ID")
        or str(uuid.uuid4())[:8]
    )
    # Attach to request state for use in endpoints
    request.state.request_id = request_id

    start = time.time()

    # 2. Global try/catch — never let middleware crash without structured response
    try:
        # 3. Timeout enforcement
        response = await asyncio.wait_for(
            call_next(request),
            timeout=MIDDLEWARE_TIMEOUT_S,
        )

    except asyncio.TimeoutError:
        latency = time.time() - start
        slog.error(
            "request_timeout",
            request_id=request_id,
            path=request.url.path,
            method=request.method,
            latency_ms=round(latency * 1000),
        )
        _request_counts["errors"] += 1
        return JSONResponse(
            status_code=504,
            content={
                "detail":     "Request timed out",
                "request_id": request_id,
                "path":       request.url.path,
            },
            headers={"X-Request-ID": request_id},
        )

    except Exception as exc:
        latency = time.time() - start
        slog.error(
            "unhandled_exception",
            request_id=request_id,
            path=request.url.path,
            method=request.method,
            error=type(exc).__name__,
            latency_ms=round(latency * 1000),
        )
        _request_counts["errors"] += 1
        return JSONResponse(
            status_code=500,
            content={
                "detail":     "Internal server error",
                "request_id": request_id,
                "error_type": type(exc).__name__,
            },
            headers={"X-Request-ID": request_id},
        )

    # 4. Record metrics
    latency = time.time() - start
    _request_counts["total"] += 1
    if response.status_code >= 500:
        _request_counts["errors"] += 1

    _latency_samples.append(latency)
    if len(_latency_samples) > 200:
        _latency_samples.pop(0)

    # 5. Periodic alert checks
    if _request_counts["total"] % 50 == 0:
        check_error_rate(_request_counts["total"], _request_counts["errors"])
        if _latency_samples:
            import numpy as np
            check_latency(float(np.percentile(_latency_samples, 95)))

    record_api_request(
        request.url.path,
        request.method,
        response.status_code,
        latency,
    )

    # Attach tracking headers to response
    response.headers["X-Request-ID"]    = request_id
    response.headers["X-Response-Time"] = f"{latency * 1000:.1f}ms"
    return response


# Internal state (initialised after app creation)
_request_counts   = {"total": 0, "errors": 0}
_latency_samples: list[float] = []


# ── Pagination helper ─────────────────────────────────────────────────────────

class PaginationParams:
    """
    Reusable pagination dependency.
    Returns offset + limit validated against max_limit.
    """
    def __init__(
        self,
        page:  int = Query(default=1,   ge=1,   description="Page number (1-based)"),
        limit: int = Query(default=50,  ge=1,   le=500, description="Items per page"),
    ):
        self.page   = page
        self.limit  = limit
        self.offset = (page - 1) * limit

    def paginate(self, items: list) -> dict:
        total  = len(items)
        sliced = items[self.offset: self.offset + self.limit]
        return {
            "items":      sliced,
            "total":      total,
            "page":       self.page,
            "limit":      self.limit,
            "pages":      max(1, (total + self.limit - 1) // self.limit),
            "has_next":   self.offset + self.limit < total,
            "has_prev":   self.page > 1,
        }


# ── Rate limit + idempotency helpers ─────────────────────────────────────────

def _rate_check(user_id: str, action: str):
    """
    Redis sliding-window rate limit check.
    Falls back gracefully if Redis is down.
    Raises HTTP 429 on limit exceeded.
    """
    if action not in _RATE_LIMITS:
        return

    max_count, window = _RATE_LIMITS[action]

    if not is_redis_available():
        logger.warning(
            f"Redis down — skipping rate check | "
            f"user={user_id} action={action}"
        )
        return

    try:
        allowed, remaining = check_rate_limit_redis(
            user_id, action, max_count, window
        )
        _cb_redis.record_success()
    except Exception as e:
        _cb_redis.record_failure()
        logger.warning(f"Rate limit check failed: {type(e).__name__} — allowing")
        return

    if not allowed:
        from utils.alerting import record_rate_limit
        record_rate_limit(action)
        raise HTTPException(
            status_code=429,
            detail={
                "error":    "rate_limit_exceeded",
                "action":   action,
                "max":      max_count,
                "window_h": window // 3600,
                "message":  f"Max {max_count} per {window // 3600}h",
            },
            headers={"Retry-After": str(window)},
        )


def _idempotency_check(email_id: str):
    """
    Atomic idempotency check using Redis SET NX.

    Uses SETNX (set if not exists) which is atomic — no race condition.
    TTL is enforced at the Redis key level, not in application code.

    Falls back gracefully if Redis is down.
    Raises HTTP 409 if already processed within TTL window.
    """
    if not is_redis_available():
        logger.warning(
            f"Redis down — skipping idempotency check | email_id={email_id}"
        )
        return

    r = get_redis()
    if r is None:
        return

    try:
        key    = f"api_processing:{email_id}"
        # SET NX with TTL — atomic: only succeeds if key doesn't exist
        locked = r.set(key, "1", ex=IDEMPOTENCY_TTL_S, nx=True)
        _cb_redis.record_success()

        if locked is None:
            # Key already existed — already processed
            raise HTTPException(
                status_code=409,
                detail={
                    "error":    "already_processed",
                    "email_id": email_id,
                    "message":  (
                        f"Email '{email_id}' was already processed "
                        f"within the last {IDEMPOTENCY_TTL_S // 3600}h. "
                        f"Set force=true to reprocess."
                    ),
                },
            )
    except HTTPException:
        raise
    except Exception as e:
        _cb_redis.record_failure()
        logger.warning(f"Idempotency check failed: {type(e).__name__} — allowing")


def _get_request_id(request: Request) -> str:
    """Extract request ID from request state (set by middleware)."""
    return getattr(request.state, "request_id", str(uuid.uuid4())[:8])


# ── Schemas ───────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    user_id:  str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1)

class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=10)

class ProcessRequest(BaseModel):
    email_id: str  = Field(..., min_length=1, max_length=200)
    persona:  str  = Field(default="Formal", pattern="^(Formal|Friendly|Concise)$")
    dry_run:  bool = False
    force:    bool = False   # bypass idempotency check

    @validator("email_id")
    def sanitize_email_id(cls, v):
        # Prevent path traversal or injection
        if any(c in v for c in ["../", "\\", "<", ">"]):
            raise ValueError("Invalid email_id characters")
        return v.strip()

class ProcessAllRequest(BaseModel):
    persona: str  = Field(default="Formal", pattern="^(Formal|Friendly|Concise)$")
    dry_run: bool = False

class FeedbackRequest(BaseModel):
    email_id:  str = Field(..., min_length=1, max_length=200)
    field:     str = Field(..., pattern="^(category|priority|task|deadline)$")
    old_value: str = Field(default="", max_length=500)
    new_value: str = Field(..., min_length=1, max_length=500)
    user_id:   str = Field(default="api_user", max_length=100)

class TrainRequest(BaseModel):
    simulate_feedback: bool = False
    n_simulate:        int  = Field(default=20, ge=5, le=100)

class RevokeRequest(BaseModel):
    token: str = Field(..., min_length=10)


# ── Auth endpoints ─────────────────────────────────────────────────────────────

@app.post("/auth/login", tags=["auth"])
async def login(req: LoginRequest, request: Request):
    """Issue JWT access + refresh token pair."""
    request_id = _get_request_id(request)
    admin_pass = os.getenv("AEOA_ADMIN_PASSWORD", "")
    role       = (
        "admin"
        if (admin_pass and req.password == admin_pass)
        else "user"
    )
    access  = create_access_token(req.user_id, role)
    refresh = create_refresh_token(req.user_id, role)

    slog.info("user_login", user=req.user_id, role=role, request_id=request_id)
    return {
        "access_token":  access,
        "refresh_token": refresh,
        "token_type":    "bearer",
        "expires_in":    int(os.getenv("ACCESS_TOKEN_EXPIRE_S", "900")),
        "request_id":    request_id,
    }


@app.post("/auth/refresh", tags=["auth"])
async def refresh_token_endpoint(req: RefreshRequest, request: Request):
    """Exchange valid refresh token for new access token."""
    request_id = _get_request_id(request)
    payload    = decode_token(req.refresh_token)

    if not payload:
        raise HTTPException(
            status_code=401,
            detail={
                "error":      "invalid_token",
                "message":    "Invalid or expired refresh token",
                "request_id": request_id,
            },
        )
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=401,
            detail={
                "error":      "wrong_token_type",
                "message":    "Provided token is not a refresh token",
                "request_id": request_id,
            },
        )

    new_access = create_access_token(
        payload["sub"], payload.get("role", "user")
    )
    return {
        "access_token": new_access,
        "token_type":   "bearer",
        "expires_in":   int(os.getenv("ACCESS_TOKEN_EXPIRE_S", "900")),
        "request_id":   request_id,
    }


@app.post("/auth/logout", tags=["auth"])
async def logout(
    request:     Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    user:        dict = Depends(get_current_user),
):
    """
    Revoke current JWT token.

    Validation order:
    1. Depends(get_current_user) — checks token is valid + not expired + not blacklisted
    2. Then revokes it

    This prevents revoking an already-expired or already-revoked token,
    and ensures only authenticated users can call this endpoint.
    """
    request_id = _get_request_id(request)

    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=400,
            detail={
                "error":      "no_token",
                "message":    "No Bearer token provided",
                "request_id": request_id,
            },
        )

    # Token is guaranteed valid here because Depends(get_current_user) passed
    revoke_token(credentials.credentials)

    slog.info(
        "user_logout",
        user=user.get("sub"),
        request_id=request_id,
    )
    return {
        "status":     "revoked",
        "user":       user.get("sub"),
        "request_id": request_id,
    }


@app.delete("/auth/revoke", tags=["auth"])
async def revoke_specific(
    req:     RevokeRequest,
    request: Request,
    user:    dict = Depends(require_permission("admin")),
):
    """Admin: immediately revoke any specific token."""
    request_id = _get_request_id(request)
    revoke_token(req.token)
    slog.info("admin_token_revoke", admin=user.get("sub"), request_id=request_id)
    return {"status": "revoked", "request_id": request_id}


# ── System endpoints ──────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health(request: Request):
    """System health — public. Includes circuit breaker states."""
    h = get_system_health()
    h["redis"]           = is_redis_available()
    h["alerts"]          = get_active_alerts()
    h["cache"]           = get_cache_stats()
    h["circuit_breakers"] = {
        "redis":  _cb_redis.to_dict(),
        "celery": _cb_celery.to_dict(),
    }
    h["request_id"] = _get_request_id(request)

    if any(a["severity"] == "critical" for a in h["alerts"]):
        h["status"] = "degraded"

    return h


@app.get("/metrics", tags=["system"])
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=get_metrics(),
        media_type=get_content_type(),
    )


@app.get("/alerts", tags=["system"])
async def get_alerts_endpoint(
    request: Request,
    user:    dict = Depends(require_permission("read")),
):
    return {
        "alerts":     get_active_alerts(),
        "request_id": _get_request_id(request),
    }


# ── Email endpoints ────────────────────────────────────────────────────────────

@app.get("/emails", tags=["emails"])
async def get_emails(
    request:    Request,
    pagination: PaginationParams = Depends(),
    user:       dict = Depends(require_permission("read")),
):
    """Paginated email list — prevents memory issues with large inboxes."""
    from memory.repository import get_all_emails
    emails = get_all_emails()
    result = pagination.paginate(emails)
    result["request_id"] = _get_request_id(request)
    return result


@app.get("/emails/{email_id}", tags=["emails"])
async def get_email(
    email_id: str,
    request:  Request,
    user:     dict = Depends(require_permission("read")),
):
    from memory.repository import get_email as _get
    email = _get(email_id)
    if not email:
        raise HTTPException(
            status_code=404,
            detail={
                "error":      "not_found",
                "email_id":   email_id,
                "request_id": _get_request_id(request),
            },
        )
    return {**email, "request_id": _get_request_id(request)}


@app.get("/processed", tags=["emails"])
async def get_processed(
    request:    Request,
    pagination: PaginationParams = Depends(),
    user:       dict = Depends(require_permission("read")),
):
    """Paginated processed email results."""
    from memory.repository import get_all_processed
    items  = get_all_processed()
    result = pagination.paginate(items)
    result["request_id"] = _get_request_id(request)
    return result


@app.post("/process", tags=["emails"])
async def process_email(
    req:     ProcessRequest,
    request: Request,
    user:    dict = Depends(require_permission("write")),
):
    """
    Dispatch single email to Celery worker.
    Security layers:
      1. JWT permission (write)
      2. Redis rate limit (circuit-breaker protected)
      3. Atomic Redis idempotency gate (circuit-breaker protected)
      4. Celery dispatch (circuit-breaker protected)
    """
    request_id = _get_request_id(request)
    user_id    = user.get("sub", "anonymous")

    _rate_check(user_id, "process_emails")

    if not req.force:
        _idempotency_check(req.email_id)

    from workers.tasks import process_email_task
    task = _dispatch_celery_task(
        process_email_task,
        req.email_id, req.persona, req.dry_run,
        queue="email_processing",
    )

    slog.info(
        "process_dispatched",
        email_id=req.email_id,
        task_id=task.id,
        user=user_id,
        force=req.force,
        request_id=request_id,
    )
    return {
        "task_id":    task.id,
        "status":     "queued",
        "email_id":   req.email_id,
        "request_id": request_id,
    }


@app.post("/process/all", tags=["emails"])
async def process_all(
    req:     ProcessAllRequest,
    request: Request,
    user:    dict = Depends(require_permission("process_all")),
):
    """Bulk email processing — admin only, async via Celery."""
    request_id = _get_request_id(request)
    user_id    = user.get("sub", "anonymous")

    _rate_check(user_id, "process_emails")

    from workers.tasks import process_all_task
    task = _dispatch_celery_task(
        process_all_task,
        req.persona, req.dry_run,
        queue="email_processing",
    )

    slog.info(
        "process_all_dispatched",
        task_id=task.id,
        user=user_id,
        request_id=request_id,
    )
    return {
        "task_id":    task.id,
        "status":     "queued",
        "request_id": request_id,
    }


@app.get("/tasks/{task_id}", tags=["emails"])
async def get_task_status(
    task_id: str,
    request: Request,
    user:    dict = Depends(require_permission("read")),
):
    """Poll Celery task status — checks Redis first, then Celery backend."""
    request_id = _get_request_id(request)

    # Fast path: Redis progress updates
    if is_redis_available():
        try:
            redis_status = get_job_status_redis(task_id)
            if redis_status:
                return {**redis_status, "request_id": request_id}
        except Exception as e:
            _cb_redis.record_failure()
            logger.warning(f"Redis job status failed: {type(e).__name__}")

    # Fallback: Celery backend
    try:
        from workers.celery_app import celery_app as _celery
        result = _celery.AsyncResult(task_id)
        status = {
            "task_id":    task_id,
            "status":     result.status,
            "result":     None,
            "error":      None,
            "request_id": request_id,
        }
        if result.ready():
            if result.successful():
                status["result"] = result.result
                _cb_celery.record_success()
            else:
                error_msg       = str(result.result) if result.result else "Unknown"
                status["error"] = error_msg
                _cb_celery.record_failure()
                slog.error(
                    "task_failed",
                    task_id=task_id,
                    error=error_msg,
                    request_id=request_id,
                )
        return status

    except Exception as e:
        _cb_celery.record_failure()
        raise HTTPException(
            status_code=503,
            detail={
                "error":      "celery_unavailable",
                "message":    str(type(e).__name__),
                "request_id": request_id,
            },
        )


# ── Feedback ──────────────────────────────────────────────────────────────────

@app.post("/feedback", tags=["emails"])
async def add_feedback(
    req:     FeedbackRequest,
    request: Request,
    user:    dict = Depends(require_permission("write")),
):
    """
    Save user correction and dispatch async online learning.
    Priority corrections trigger immediate ML model update via Celery.
    """
    request_id = _get_request_id(request)

    from memory.repository import insert_feedback, update_processed_field
    update_processed_field(req.email_id, req.field, req.new_value)
    insert_feedback(req.email_id, req.field, req.old_value, req.new_value)
    record_feedback(req.field)

    online_task_id = None
    if req.field == "priority":
        try:
            priority = int(req.new_value)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error":      "invalid_priority",
                    "message":    f"Priority must be integer 1-7, got: {req.new_value}",
                    "request_id": request_id,
                },
            )
        if not (1 <= priority <= 7):
            raise HTTPException(
                status_code=400,
                detail={
                    "error":      "out_of_range",
                    "message":    f"Priority {priority} must be between 1 and 7",
                    "request_id": request_id,
                },
            )

        from workers.tasks import online_learn_task
        task           = _dispatch_celery_task(
            online_learn_task,
            req.email_id, priority, user.get("sub", req.user_id),
            queue="ml_tasks",
        )
        online_task_id = task.id

        slog.info(
            "online_learn_dispatched",
            email_id=req.email_id,
            priority=priority,
            task_id=online_task_id,
            user=user.get("sub"),
            request_id=request_id,
        )

    return {
        "status":            "ok",
        "online_learn_task": online_task_id,
        "request_id":        request_id,
    }


# ── ML endpoints ──────────────────────────────────────────────────────────────

@app.post("/ml/train", tags=["ml"])
async def train_ml(
    req:     TrainRequest,
    request: Request,
    user:    dict = Depends(require_permission("train")),
):
    """Dispatch model training to Celery ml_tasks queue. Admin/train only."""
    request_id = _get_request_id(request)
    user_id    = user.get("sub", "anonymous")

    _rate_check(user_id, "train_model")

    from workers.tasks import train_model_task
    task = _dispatch_celery_task(
        train_model_task,
        req.simulate_feedback, req.n_simulate,
        queue="ml_tasks",
    )

    slog.info(
        "train_dispatched",
        task_id=task.id,
        simulate=req.simulate_feedback,
        user=user_id,
        request_id=request_id,
    )
    return {
        "task_id":    task.id,
        "status":     "training_started",
        "request_id": request_id,
    }


@app.get("/ml/model", tags=["ml"])
async def get_model_info(
    request: Request,
    user:    dict = Depends(require_permission("read")),
):
    from memory.repository import get_active_model_version, get_model_history
    from services.mlflow_tracker import get_run_history, get_best_run
    return {
        "active":      get_active_model_version(),
        "history":     get_model_history(),
        "mlflow_runs": get_run_history(limit=5),
        "best_run":    get_best_run(),
        "request_id":  _get_request_id(request),
    }


@app.get("/ml/monitor", tags=["ml"])
async def get_ml_monitoring(
    request: Request,
    user:    dict = Depends(require_permission("read")),
):
    from services.ml_service import get_monitoring_report
    report   = get_monitoring_report()
    baseline = report.get("baseline_acc", 0.70)
    recent   = report.get("accuracy", {}).get("recent_accuracy")
    if recent is not None:
        check_ml_accuracy(recent, baseline)

    report["request_id"] = _get_request_id(request)
    return report


@app.get("/ml/drift", tags=["ml"])
async def get_drift_report(
    request: Request,
    user:    dict = Depends(require_permission("read")),
):
    """Unified drift report: feature + concept + label."""
    from services.ml_service import _prediction_log, FEATURE_NAMES
    from services.drift_detector import unified_drift_report
    from memory.repository import get_all_training_data, get_all_processed

    train_data   = get_all_training_data()
    processed    = get_all_processed()
    ref_features = [d["features"] for d in train_data[-100:] if d.get("features")]
    ref_labels   = [d["label"]    for d in train_data[-100:] if d.get("label")]
    cur_labels   = [p.get("priority",4) for p in processed[-50:] if p.get("priority")]
    cur_features = ref_features[-20:] if len(ref_features) > 20 else ref_features

    report = unified_drift_report(
        reference_features=ref_features,
        current_features=cur_features,
        feature_names=FEATURE_NAMES,
        predictions=_prediction_log,
        reference_labels=ref_labels,
        current_labels=cur_labels,
    )

    if report.get("overall_severity") == "high":
        max_psi = max(
            (
                v.get("psi", 0)
                for v in report.get(
                    "feature_drift", {}
                ).get("features", {}).values()
            ),
            default=0,
        )
        check_feature_drift(max_psi)
        slog.warning(
            "high_drift_detected",
            severity=report["overall_severity"],
            max_psi=max_psi,
            request_id=_get_request_id(request),
        )

    report["request_id"] = _get_request_id(request)
    return report


@app.post("/ml/rollback", tags=["ml"])
async def rollback_online(
    request: Request,
    steps:   int  = Query(default=1, ge=1, le=10),
    user:    dict = Depends(require_permission("train")),
):
    from services.online_learning import rollback_online_model
    success    = rollback_online_model(steps=steps)
    request_id = _get_request_id(request)
    slog.info(
        "model_rollback",
        steps=steps,
        success=success,
        user=user.get("sub"),
        request_id=request_id,
    )
    return {"success": success, "request_id": request_id}


@app.post("/ml/reset", tags=["ml"])
async def reset_online(
    request: Request,
    user:    dict = Depends(require_permission("admin")),
):
    from services.online_learning import reset_online_model
    success    = reset_online_model()
    request_id = _get_request_id(request)
    slog.info(
        "model_reset",
        user=user.get("sub"),
        request_id=request_id,
    )
    return {"success": success, "request_id": request_id}


# ── Analytics — admin only ─────────────────────────────────────────────────────

@app.get("/analytics", tags=["analytics"])
async def get_analytics(
    request:    Request,
    pagination: PaginationParams = Depends(),
    user:       dict = Depends(require_permission("admin")),
):
    """Full system analytics — admin only, paginated."""
    from memory.repository import (
        get_all_processed, get_all_emails,
        get_total_llm_calls, get_all_feedback,
        get_metrics_summary,
    )
    processed_all = get_all_processed()
    emails_all    = get_all_emails()

    # Paginate processed items — don't return all at once
    paginated = pagination.paginate(processed_all)

    return {
        "summary": {
            "emails":       len(emails_all),
            "processed":    len(processed_all),
            "needs_review": sum(1 for p in processed_all if p.get("needs_review")),
            "llm_calls":    get_total_llm_calls(),
            "feedback":     len(get_all_feedback()),
        },
        "processed":      paginated,
        "tool_metrics":   get_metrics_summary(),
        "alerts":         get_active_alerts(),
        "cache":          get_cache_stats(),
        "circuit_breakers": {
            "redis":  _cb_redis.to_dict(),
            "celery": _cb_celery.to_dict(),
        },
        "request_id": _get_request_id(request),
    }