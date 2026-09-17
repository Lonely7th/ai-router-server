from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from .metrics import UPSTREAM_ERRORS
from .schemas import TokenUsage


@dataclass(slots=True)
class ProviderError(Exception):
    status_code: int
    message: str
    retry_after: str | None = None

    def __str__(self) -> str:
        return self.message


def _provider_error_message(body: bytes, status_code: int) -> str:
    try:
        parsed = json.loads(body)
        value = parsed.get("error", parsed) if isinstance(parsed, dict) else None
        if isinstance(value, dict) and isinstance(value.get("message"), str):
            return value["message"][:500]
        if isinstance(value, str):
            return value[:500]
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    return f"AI provider request failed with HTTP {status_code}"


class SSEUsageParser:
    def __init__(self) -> None:
        self._buffer = b""
        self.usage = TokenUsage()

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        while b"\n" in self._buffer:
            raw_line, self._buffer = self._buffer.split(b"\n", 1)
            self._parse_line(raw_line.rstrip(b"\r"))

    def finish(self) -> None:
        if self._buffer:
            self._parse_line(self._buffer.rstrip(b"\r"))
            self._buffer = b""

    def _parse_line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            return
        try:
            payload: Any = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if isinstance(payload, dict) and payload.get("usage") is not None:
            self.usage = TokenUsage.from_provider(payload["usage"])


class DeepSeekClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        connect_timeout: float,
        request_timeout: float,
        initial_retries: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=request_timeout,
            write=30.0,
            pool=connect_timeout,
        )
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "User-Agent": "jj-office-ai-service/0.1.0",
            },
            timeout=timeout,
            limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
            transport=transport,
        )
        self.initial_retries = initial_retries

    async def send(self, payload: dict[str, Any]) -> httpx.Response:
        attempts = self.initial_retries + 1
        for attempt in range(attempts):
            request = self.client.build_request("POST", "/chat/completions", json=payload)
            try:
                response = await self.client.send(request, stream=True)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
                if attempt + 1 < attempts:
                    await asyncio.sleep(0.25 * (2**attempt))
                    continue
                UPSTREAM_ERRORS.labels(status_code="connection").inc()
                raise ProviderError(502, "Unable to connect to the AI provider") from exc
            if response.status_code < 400:
                return response
            body = await response.aread()
            retry_after = response.headers.get("retry-after")
            await response.aclose()
            if response.status_code in {429, 500, 502, 503, 504} and attempt + 1 < attempts:
                await asyncio.sleep(0.25 * (2**attempt))
                continue
            UPSTREAM_ERRORS.labels(status_code=str(response.status_code)).inc()
            if response.status_code in {401, 402, 403}:
                # Provider credentials and account balance are service concerns. Returning the
                # upstream status would incorrectly tell the desktop client that its user token
                # is invalid and could expose provider-account details.
                raise ProviderError(503, "AI provider is temporarily unavailable", retry_after)
            public_status = response.status_code if response.status_code in {400, 422, 429} else 502
            raise ProviderError(
                public_status,
                _provider_error_message(body, response.status_code),
                retry_after,
            )
        raise ProviderError(502, "AI provider request failed")

    @staticmethod
    async def stream_bytes(
        response: httpx.Response, parser: SSEUsageParser
    ) -> AsyncIterator[bytes]:
        # aiter_bytes decodes any upstream content encoding. We deliberately do not forward
        # Content-Encoding, so yielding raw compressed bytes here would corrupt the client stream.
        async for chunk in response.aiter_bytes():
            parser.feed(chunk)
            yield chunk

    @staticmethod
    async def read_json(response: httpx.Response) -> tuple[dict[str, Any], TokenUsage]:
        try:
            body = await response.aread()
            parsed = json.loads(body)
            if not isinstance(parsed, dict):
                raise ValueError("provider response is not an object")
            return parsed, TokenUsage.from_provider(parsed.get("usage"))
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise ProviderError(502, "AI provider returned an invalid JSON response") from exc
        finally:
            await response.aclose()

    async def close(self) -> None:
        await self.client.aclose()
