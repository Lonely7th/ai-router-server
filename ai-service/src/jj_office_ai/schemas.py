from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: Any = None


class ChatCompletionRequest(BaseModel):
    """Supported, safe subset of the OpenAI Chat Completions request."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    stop: str | list[str] | None = None
    response_format: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    thinking: dict[str, Any] | None = None
    seed: int | None = None
    n: int | None = Field(default=None, ge=1, le=1)

    @model_validator(mode="after")
    def only_one_output_limit(self) -> ChatCompletionRequest:
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("Set either max_tokens or max_completion_tokens, not both")
        if self.thinking is not None and self.reasoning_effort is not None:
            raise ValueError("Set either thinking or reasoning_effort, not both")
        return self

    @property
    def requested_output_tokens(self) -> int | None:
        return self.max_tokens or self.max_completion_tokens

    def upstream_payload(
        self, *, model: str, max_output_tokens: int, user_id: str
    ) -> dict[str, Any]:
        payload = self.model_dump(
            exclude_none=True,
            exclude={
                "model",
                "max_tokens",
                "max_completion_tokens",
                # OpenAI compatibility fields currently unsupported by DeepSeek.
                "parallel_tool_calls",
                "seed",
                "n",
            },
        )
        payload["model"] = model
        payload["max_tokens"] = max_output_tokens
        payload["user_id"] = user_id
        for message in payload["messages"]:
            if message.get("role") == "developer":
                message["role"] = "system"
        if self.stream:
            stream_options = dict(payload.get("stream_options") or {})
            stream_options["include_usage"] = True
            payload["stream_options"] = stream_options
        else:
            payload.pop("stream_options", None)
        return payload


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_provider(cls, value: Any) -> TokenUsage:
        if not isinstance(value, dict):
            return cls()
        prompt = int(value.get("prompt_tokens") or 0)
        completion = int(value.get("completion_tokens") or 0)
        total = int(value.get("total_tokens") or prompt + completion)
        return cls(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
