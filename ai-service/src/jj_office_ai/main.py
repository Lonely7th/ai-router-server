from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import make_asgi_app

from . import __version__
from .auth import AuthContext, authenticate
from .config import Settings, get_settings
from .deepseek import DeepSeekClient, ProviderError, SSEUsageParser
from .logging_config import configure_logging
from .metrics import ACTIVE_REQUESTS, REQUEST_DURATION, REQUESTS, observe_tokens
from .quota import QuotaUnavailable, Reservation, create_quota_client
from .rate_limit import RateLimitExceeded, RateLimitLease, create_rate_limiter
from .schemas import ChatCompletionRequest, ErrorDetail, ErrorResponse, TokenUsage

logger = logging.getLogger(__name__)
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

MODEL_ALIASES = {
    "fast": "deepseek-flash",
    "pro": "deepseek-v4-pro",
    "deepseek-v4-flash": "deepseek-flash",
    "deepseek-v4-flash-vision-exp": "deepseek-flash",
    "deepseek-chat": "deepseek-flash",
    "deepseek-reasoner": "deepseek-flash",
}


def _error(
    status_code: int, message: str, error_type: str, code: str | None = None
) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(message=message, type=error_type, code=code))
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _request_id(request: Request) -> str:
    supplied = request.headers.get("x-request-id", "")
    if REQUEST_ID_RE.fullmatch(supplied):
        return supplied
    return uuid.uuid4().hex


def _resolve_model(requested: str | None, settings: Settings) -> str:
    value = requested or settings.default_model
    if value == "default":
        value = settings.default_model
    value = MODEL_ALIASES.get(value, value)
    if value not in settings.parsed_allowed_models:
        raise HTTPException(status_code=400, detail=f"Model '{requested}' is not available")
    return value


def _max_output_tokens(body: ChatCompletionRequest, settings: Settings) -> int:
    requested = body.requested_output_tokens or settings.max_output_tokens
    if requested > settings.max_output_tokens:
        raise HTTPException(
            status_code=400,
            detail=(
                "Requested output exceeds the service limit of "
                f"{settings.max_output_tokens} tokens"
            ),
        )
    return requested


async def _reserve(
    request: Request,
    *,
    request_id: str,
    auth: AuthContext,
    model: str,
    max_tokens: int,
    request_bytes: int,
) -> Reservation:
    try:
        return await request.app.state.quota.reserve(
            request_id=request_id,
            user_id=auth.user_id,
            model=model,
            max_output_tokens=max_tokens,
            request_bytes=request_bytes,
        )
    except QuotaUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _response_headers(request_id: str) -> dict[str, str]:
    return {
        "X-Request-ID": request_id,
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
    }


def create_app(
    settings: Settings | None = None,
    *,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.settings = resolved_settings
        application.state.deepseek = DeepSeekClient(
            api_key=resolved_settings.deepseek_api_key.get_secret_value(),
            base_url=resolved_settings.deepseek_base_url,
            connect_timeout=resolved_settings.connect_timeout_seconds,
            request_timeout=resolved_settings.request_timeout_seconds,
            initial_retries=resolved_settings.upstream_initial_retries,
            transport=upstream_transport,
        )
        application.state.rate_limiter = create_rate_limiter(
            redis_url=resolved_settings.redis_url.get_secret_value(),
            requests_per_minute=resolved_settings.rate_limit_requests_per_minute,
            max_concurrent=resolved_settings.max_concurrent_requests_per_user,
            request_timeout=resolved_settings.request_timeout_seconds,
        )
        await application.state.rate_limiter.ready()
        application.state.quota = create_quota_client(
            resolved_settings.quota_service_url,
            resolved_settings.quota_service_token.get_secret_value(),
        )
        logger.info(
            "service_started environment=%s auth_mode=%s models=%s quota=%s redis=%s",
            resolved_settings.environment,
            resolved_settings.auth_mode,
            ",".join(resolved_settings.parsed_allowed_models),
            bool(resolved_settings.quota_service_url),
            bool(resolved_settings.redis_url.get_secret_value()),
        )
        try:
            yield
        finally:
            await application.state.deepseek.close()
            await application.state.rate_limiter.close()
            await application.state.quota.close()
            logger.info("service_stopped")

    application = FastAPI(
        title="JJ Office AI Service",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if resolved_settings.environment != "production" else None,
        redoc_url=None,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.parsed_cors_allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Request-ID",
            "X-User-Id",
        ],
        expose_headers=["X-Request-ID", "Retry-After"],
        max_age=86400,
    )

    @application.middleware("http")
    async def request_guard(request: Request, call_next: Any) -> Any:
        request.state.request_id = _request_id(request)
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > resolved_settings.max_request_bytes:
                    return _error(413, "Request body is too large", "request_too_large")
            except ValueError:
                return _error(400, "Invalid Content-Length header", "invalid_request_error")
        if request.method == "POST" and request.url.path == "/v1/chat/completions":
            # Content-Length is not guaranteed (for example, chunked HTTP requests). Reading
            # here lets Starlette cache the body for FastAPI while enforcing the real byte size.
            raw_body = await request.body()
            if len(raw_body) > resolved_settings.max_request_bytes:
                return _error(413, "Request body is too large", "request_too_large")
        response = await call_next(request)
        response.headers.setdefault("X-Request-ID", request.state.request_id)
        return response

    @application.exception_handler(HTTPException)
    async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
        message = str(exc.detail) if exc.detail else "Request failed"
        response = _error(exc.status_code, message, "invalid_request_error")
        if exc.headers:
            response.headers.update(exc.headers)
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(item) for item in first.get("loc", ()) if item != "body")
        message = str(first.get("msg", "Invalid request"))
        if location:
            message = f"{location}: {message}"
        return _error(422, message, "invalid_request_error")

    @application.exception_handler(ProviderError)
    async def provider_error_handler(_: Request, exc: ProviderError) -> JSONResponse:
        response = _error(exc.status_code, exc.message, "upstream_error")
        if exc.retry_after:
            response.headers["Retry-After"] = exc.retry_after
        return response

    @application.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "unhandled_request_error request_id=%s error=%s",
            request.state.request_id,
            type(exc).__name__,
        )
        return _error(500, "Internal server error", "server_error")

    @application.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"service": resolved_settings.service_name, "version": __version__}

    @application.get("/health/live", tags=["health"])
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/health/ready", tags=["health"], response_model=None)
    async def ready() -> JSONResponse | dict[str, str]:
        if not resolved_settings.ready:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "missing_api_key"},
            )
        return {"status": "ready"}

    @application.get("/v1/models", tags=["ai"])
    async def models(_: AuthContext = Depends(authenticate)) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": model, "object": "model", "owned_by": "deepseek"}
                for model in resolved_settings.parsed_allowed_models
            ],
        }

    @application.post("/v1/chat/completions", tags=["ai"], response_model=None)
    async def chat_completions(
        request: Request,
        body: ChatCompletionRequest,
        auth: AuthContext = Depends(authenticate),
    ) -> JSONResponse | StreamingResponse:
        request_id = request.state.request_id
        request_bytes = len(body.model_dump_json().encode("utf-8"))
        if request_bytes > resolved_settings.max_request_bytes:
            raise HTTPException(status_code=413, detail="Request body is too large")
        if not resolved_settings.ready:
            raise HTTPException(status_code=503, detail="AI provider is not configured")

        model = _resolve_model(body.model, resolved_settings)
        max_tokens = _max_output_tokens(body, resolved_settings)
        try:
            lease: RateLimitLease = await request.app.state.rate_limiter.acquire(auth.user_id)
        except RateLimitExceeded as exc:
            response = _error(429, exc.reason, "rate_limit_error")
            response.headers["Retry-After"] = str(exc.retry_after)
            return response

        reservation: Reservation | None = None
        upstream: httpx.Response | None = None
        stream_label = "true" if body.stream else "false"
        started = time.monotonic()
        ACTIVE_REQUESTS.labels(model=model, stream=stream_label).inc()

        try:
            reservation = await _reserve(
                request,
                request_id=request_id,
                auth=auth,
                model=model,
                max_tokens=max_tokens,
                request_bytes=request_bytes,
            )
            payload = body.upstream_payload(
                model=model,
                max_output_tokens=max_tokens,
                user_id=auth.provider_user_id,
            )
            if (
                body.model == "deepseek-chat"
                and body.thinking is None
                and body.reasoning_effort is None
            ):
                payload["thinking"] = {"type": "disabled"}
            elif (
                body.model == "deepseek-reasoner"
                and body.thinking is None
                and body.reasoning_effort is None
            ):
                payload["thinking"] = {"type": "enabled"}
            upstream = await request.app.state.deepseek.send(payload)
        except Exception:
            if reservation is not None:
                await request.app.state.quota.release(
                    reservation, request_id=request_id, reason="upstream_not_started"
                )
            await lease.release()
            ACTIVE_REQUESTS.labels(model=model, stream=stream_label).dec()
            REQUESTS.labels(model=model, stream=stream_label, status="failed_before_stream").inc()
            REQUEST_DURATION.labels(model=model, stream=stream_label).observe(
                time.monotonic() - started
            )
            raise

        if not body.stream:
            try:
                parsed, usage = await request.app.state.deepseek.read_json(upstream)
                observe_tokens(model, usage.prompt_tokens, usage.completion_tokens)
                await request.app.state.quota.finalize(
                    reservation,
                    request_id=request_id,
                    model=model,
                    usage=usage,
                    outcome="completed",
                )
                REQUESTS.labels(model=model, stream=stream_label, status="completed").inc()
                return JSONResponse(content=parsed, headers=_response_headers(request_id))
            except Exception:
                await request.app.state.quota.finalize(
                    reservation,
                    request_id=request_id,
                    model=model,
                    usage=TokenUsage(),
                    outcome="invalid_upstream_response",
                )
                REQUESTS.labels(model=model, stream=stream_label, status="failed").inc()
                raise
            finally:
                await lease.release()
                ACTIVE_REQUESTS.labels(model=model, stream=stream_label).dec()
                REQUEST_DURATION.labels(model=model, stream=stream_label).observe(
                    time.monotonic() - started
                )

        parser = SSEUsageParser()

        async def relay() -> AsyncIterator[bytes]:
            completed = False
            outcome = "stream_failed"
            try:
                async for chunk in request.app.state.deepseek.stream_bytes(upstream, parser):
                    yield chunk
                parser.finish()
                completed = True
                outcome = "completed" if parser.usage.total_tokens else "completed_without_usage"
                observe_tokens(
                    model, parser.usage.prompt_tokens, parser.usage.completion_tokens
                )
                await request.app.state.quota.finalize(
                    reservation,
                    request_id=request_id,
                    model=model,
                    usage=parser.usage,
                    outcome=outcome,
                )
                REQUESTS.labels(model=model, stream=stream_label, status=outcome).inc()
            except asyncio.CancelledError:
                outcome = "client_disconnected"
                REQUESTS.labels(model=model, stream=stream_label, status=outcome).inc()
                raise
            except Exception:
                REQUESTS.labels(model=model, stream=stream_label, status=outcome).inc()
                logger.exception("stream_failed request_id=%s model=%s", request_id, model)
                raise
            finally:
                if not completed:
                    await request.app.state.quota.finalize(
                        reservation,
                        request_id=request_id,
                        model=model,
                        usage=parser.usage,
                        outcome=outcome,
                    )
                await upstream.aclose()
                await lease.release()
                ACTIVE_REQUESTS.labels(model=model, stream=stream_label).dec()
                REQUEST_DURATION.labels(model=model, stream=stream_label).observe(
                    time.monotonic() - started
                )
                logger.info(
                    "ai_request request_id=%s user_hash=%s model=%s stream=true "
                    "outcome=%s input_tokens=%s output_tokens=%s",
                    request_id,
                    auth.provider_user_id[:12],
                    model,
                    outcome,
                    parser.usage.prompt_tokens,
                    parser.usage.completion_tokens,
                )

        content_type = upstream.headers.get("content-type", "text/event-stream").split(";", 1)[0]
        return StreamingResponse(
            relay(),
            status_code=upstream.status_code,
            media_type=content_type,
            headers=_response_headers(request_id),
        )

    application.mount("/metrics", make_asgi_app())
    return application


app = create_app()
