from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import httpx

from .metrics import QUOTA_ERRORS
from .schemas import TokenUsage

logger = logging.getLogger(__name__)


class QuotaUnavailable(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Reservation:
    reservation_id: str | None


class QuotaClient(Protocol):
    async def reserve(
        self,
        *,
        request_id: str,
        user_id: str,
        model: str,
        max_output_tokens: int,
        request_bytes: int,
    ) -> Reservation: ...

    async def finalize(
        self,
        reservation: Reservation,
        *,
        request_id: str,
        model: str,
        usage: TokenUsage,
        outcome: str,
    ) -> None: ...

    async def release(
        self, reservation: Reservation, *, request_id: str, reason: str
    ) -> None: ...

    async def close(self) -> None: ...


class DisabledQuotaClient:
    async def reserve(self, **_: object) -> Reservation:
        return Reservation(None)

    async def finalize(self, reservation: Reservation, **_: object) -> None:
        return None

    async def release(self, reservation: Reservation, **_: object) -> None:
        return None

    async def close(self) -> None:
        return None


class HttpQuotaClient:
    """CloudBase quota API client.

    Reservations are fail-closed. Finalization failures are logged because the response may
    already have been streamed to the client; the future billing service must also reconcile
    reservations that remain open.
    """

    def __init__(self, base_url: str, token: str) -> None:
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(5.0, connect=3.0),
        )

    async def reserve(
        self,
        *,
        request_id: str,
        user_id: str,
        model: str,
        max_output_tokens: int,
        request_bytes: int,
    ) -> Reservation:
        try:
            response = await self.client.post(
                "/v1/internal/ai/reservations",
                json={
                    "request_id": request_id,
                    "user_id": user_id,
                    "model": model,
                    "max_output_tokens": max_output_tokens,
                    "request_bytes": request_bytes,
                },
            )
            response.raise_for_status()
            reservation_id = response.json().get("reservation_id")
            if not isinstance(reservation_id, str) or not reservation_id:
                raise ValueError("quota service returned no reservation_id")
            return Reservation(reservation_id)
        except (httpx.HTTPError, ValueError) as exc:
            QUOTA_ERRORS.labels(operation="reserve").inc()
            logger.warning(
                "quota_reserve_failed request_id=%s error=%s",
                request_id,
                type(exc).__name__,
            )
            raise QuotaUnavailable("Quota service is temporarily unavailable") from exc

    async def finalize(
        self,
        reservation: Reservation,
        *,
        request_id: str,
        model: str,
        usage: TokenUsage,
        outcome: str,
    ) -> None:
        if not reservation.reservation_id:
            return
        try:
            response = await self.client.post(
                f"/v1/internal/ai/reservations/{reservation.reservation_id}/finalize",
                json={
                    "request_id": request_id,
                    "model": model,
                    "usage": usage.model_dump(),
                    "outcome": outcome,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            QUOTA_ERRORS.labels(operation="finalize").inc()
            logger.error(
                "quota_finalize_failed request_id=%s reservation_id=%s error=%s",
                request_id,
                reservation.reservation_id,
                type(exc).__name__,
            )

    async def release(
        self, reservation: Reservation, *, request_id: str, reason: str
    ) -> None:
        if not reservation.reservation_id:
            return
        try:
            response = await self.client.post(
                f"/v1/internal/ai/reservations/{reservation.reservation_id}/release",
                json={"request_id": request_id, "reason": reason},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            QUOTA_ERRORS.labels(operation="release").inc()
            logger.error(
                "quota_release_failed request_id=%s reservation_id=%s error=%s",
                request_id,
                reservation.reservation_id,
                type(exc).__name__,
            )

    async def close(self) -> None:
        await self.client.aclose()


def create_quota_client(base_url: str, token: str) -> QuotaClient:
    if not base_url:
        return DisabledQuotaClient()
    return HttpQuotaClient(base_url, token)
