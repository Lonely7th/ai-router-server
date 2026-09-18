from __future__ import annotations

import json
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from jj_office_ai.config import Settings
from jj_office_ai.main import create_app
from jj_office_ai.rate_limit import MemoryRateLimiter, RateLimitExceeded


def settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "AI_ENVIRONMENT": "test",
        "AI_AUTH_MODE": "none",
        "DEEPSEEK_API_KEY": "test-provider-key",
        "AI_ALLOWED_MODELS": "deepseek-flash,deepseek-v4-pro",
        "AI_DEFAULT_MODEL": "deepseek-flash",
        "AI_UPSTREAM_INITIAL_RETRIES": 0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def completion(*, stream: bool = False, model: str = "deepseek-flash") -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Improve this paragraph."}],
        "max_tokens": 512,
        "stream": stream,
    }


def test_health_and_models() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("health checks must not call the provider")

    app = create_app(settings(), upstream_transport=httpx.MockTransport(handler))
    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "ok"}
        assert client.get("/health/ready").json() == {"status": "ready"}
        models = client.get("/v1/models").json()["data"]
        assert [item["id"] for item in models] == ["deepseek-flash", "deepseek-v4-pro"]


def test_ready_fails_without_provider_key() -> None:
    app = create_app(settings(DEEPSEEK_API_KEY=""))
    with TestClient(app) as client:
        response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["reason"] == "missing_api_key"


def test_non_streaming_request_is_sanitized_and_forwarded() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer test-provider-key"
        return httpx.Response(
            200,
            json={
                "id": "chat-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Done"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "prompt_cache_hit_tokens": 6,
                    "prompt_cache_miss_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 11,
                },
            },
        )

    app = create_app(settings(), upstream_transport=httpx.MockTransport(handler))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"X-User-Id": "local-user", "X-Request-ID": "request-123"},
            json=completion(model="deepseek-chat"),
        )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "request-123"
    assert response.json()["choices"][0]["message"]["content"] == "Done"
    assert captured["model"] == "deepseek-flash"
    assert captured["max_tokens"] == 512
    assert captured["thinking"] == {"type": "disabled"}
    assert captured["user_id"] != "local-user"
    assert len(captured["user_id"]) == 40


def test_streaming_request_preserves_sse_and_forces_usage() -> None:
    captured: dict[str, Any] = {}
    event = (
        b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}],'
        b'"usage":null}\n\n'
        b'data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=event, headers={"content-type": "text/event-stream"})

    app = create_app(settings(), upstream_transport=httpx.MockTransport(handler))
    with TestClient(app) as client:
        with client.stream(
            "POST", "/v1/chat/completions", json=completion(stream=True)
        ) as response:
            body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert body == event
    assert captured["stream_options"] == {"include_usage": True}


def test_static_auth_and_validation_errors_are_openai_shaped() -> None:
    app = create_app(settings(AI_AUTH_MODE="static", AI_STATIC_TOKENS="test-client-token"))
    with TestClient(app) as client:
        missing = client.get("/v1/models")
        wrong_model = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer test-client-token",
                "X-User-Id": "user-1",
            },
            json=completion(model="not-allowed"),
        )

    assert missing.status_code == 401
    assert missing.json()["error"]["message"] == "Missing bearer token"
    assert wrong_model.status_code == 400
    assert wrong_model.json()["error"]["type"] == "invalid_request_error"


def test_static_user_quota_reservation_and_settlement_contract() -> None:
    quota_calls: list[dict[str, Any]] = []

    async def quota_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        quota_calls.append(
            {
                "path": request.url.path,
                "authorization": request.headers.get("authorization"),
                "body": body,
            }
        )
        if request.url.path.endswith("/finalize"):
            return httpx.Response(200, json={"finalized": True, "charged_tokens": 11})
        return httpx.Response(
            200,
            json={"reservation_id": "air_test_reservation_1234567890", "reserved_tokens": 520},
        )

    async def provider_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chat-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "AI服务连接成功"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "prompt_cache_hit_tokens": 6,
                    "prompt_cache_miss_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 11,
                },
            },
        )

    app = create_app(
        settings(
            AI_AUTH_MODE="static",
            AI_STATIC_TOKENS="test-client-token",
            AI_QUOTA_SERVICE_URL="https://quota.example/chat-router-server",
            AI_QUOTA_SERVICE_TOKEN="internal-service-token",
        ),
        upstream_transport=httpx.MockTransport(provider_handler),
        quota_transport=httpx.MockTransport(quota_handler),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer test-client-token",
                "X-User-Id": "local-test-user",
            },
            json=completion(),
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "AI服务连接成功"
    assert [call["path"] for call in quota_calls] == [
        "/chat-router-server/v1/internal/ai/reservations",
        "/chat-router-server/v1/internal/ai/reservations/air_test_reservation_1234567890/finalize",
    ]
    assert all(call["authorization"] == "Bearer internal-service-token" for call in quota_calls)
    assert quota_calls[0]["body"]["user_id"] == "local-test-user"
    assert quota_calls[1]["body"]["usage"] == {
        "prompt_tokens": 9,
        "prompt_cache_hit_tokens": 6,
        "prompt_cache_miss_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 11,
    }


def test_quota_exhaustion_is_returned_as_payment_required() -> None:
    async def quota_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            402,
            json={"error": {"code": "QUOTA_EXCEEDED", "message": "AI 额度不足。"}},
        )

    async def provider_handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("provider must not be called when quota is exhausted")

    app = create_app(
        settings(
            AI_AUTH_MODE="static",
            AI_STATIC_TOKENS="test-client-token",
            AI_QUOTA_SERVICE_URL="https://quota.example/chat-router-server",
            AI_QUOTA_SERVICE_TOKEN="internal-service-token",
        ),
        upstream_transport=httpx.MockTransport(provider_handler),
        quota_transport=httpx.MockTransport(quota_handler),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer test-client-token",
                "X-User-Id": "local-test-user",
            },
            json=completion(),
        )

    assert response.status_code == 402
    assert response.json()["error"]["message"] == "AI 额度不足。"


def test_upstream_credentials_failure_is_not_exposed_as_user_auth_failure() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "provider secret is invalid"}})

    app = create_app(settings(), upstream_transport=httpx.MockTransport(handler))
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/chat/completions", json=completion())

    assert response.status_code == 503
    assert response.json()["error"]["message"] == "AI provider is temporarily unavailable"


def test_request_size_limit_uses_actual_body_size() -> None:
    app = create_app(settings(AI_MAX_REQUEST_BYTES=1024))
    payload = completion()
    payload["messages"][0]["content"] = "x" * 1500
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 413


def test_openai_compatibility_fields_are_normalized() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    payload = completion()
    payload.pop("max_tokens")
    payload["max_completion_tokens"] = 321
    payload["messages"] = [{"role": "developer", "content": "Be concise."}]
    payload["parallel_tool_calls"] = True
    payload["seed"] = 7
    payload["n"] = 1
    payload["stream_options"] = {"include_usage": False}

    app = create_app(settings(), upstream_transport=httpx.MockTransport(handler))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert captured["max_tokens"] == 321
    assert captured["messages"][0]["role"] == "system"
    assert "max_completion_tokens" not in captured
    assert "parallel_tool_calls" not in captured
    assert "seed" not in captured
    assert "n" not in captured
    assert "stream_options" not in captured


def test_cors_preflight_supports_electron_web_requests() -> None:
    app = create_app(settings())
    with TestClient(app) as client:
        response = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "app://jj-office",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type,x-user-id",
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


def test_production_requires_jwt_authentication() -> None:
    with pytest.raises(ValueError, match="AI_AUTH_MODE=jwt"):
        settings(AI_ENVIRONMENT="production", AI_AUTH_MODE="static", AI_STATIC_TOKENS="token")


def test_jwt_authentication_validates_signature_issuer_and_audience() -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    token = jwt.encode(
        {
            "sub": "wechat-user-1",
            "iat": 1_788_000_000,
            "exp": 1_800_000_000,
            "iss": "jj-office-cloudbase",
            "aud": "jj-office-ai",
        },
        private_key,
        algorithm="RS256",
    )
    app = create_app(settings(AI_AUTH_MODE="jwt", AI_JWT_PUBLIC_KEY_PEM=public_pem))

    with TestClient(app) as client:
        accepted = client.get("/v1/models", headers={"Authorization": f"Bearer {token}"})
        rejected = client.get("/v1/models", headers={"Authorization": f"Bearer {token}tampered"})

    assert accepted.status_code == 200
    assert rejected.status_code == 401


@pytest.mark.asyncio
async def test_memory_rate_limiter_enforces_concurrency_and_releases() -> None:
    limiter = MemoryRateLimiter(requests_per_minute=10, max_concurrent=1)
    lease = await limiter.acquire("user-1")
    with pytest.raises(RateLimitExceeded, match="Concurrent"):
        await limiter.acquire("user-1")
    await lease.release()
    second_lease = await limiter.acquire("user-1")
    await second_lease.release()
