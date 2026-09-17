from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    service_name: str = Field(default="jj-office-ai-service", alias="AI_SERVICE_NAME")
    environment: Literal["development", "test", "production"] = Field(
        default="development", alias="AI_ENVIRONMENT"
    )
    log_level: str = Field(default="INFO", alias="AI_LOG_LEVEL")
    cors_allowed_origins: str = Field(default="*", alias="AI_CORS_ALLOWED_ORIGINS")

    auth_mode: Literal["none", "static", "jwt"] = Field(default="none", alias="AI_AUTH_MODE")
    static_tokens: str = Field(default="", alias="AI_STATIC_TOKENS")
    jwt_public_key_file: str = Field(default="", alias="AI_JWT_PUBLIC_KEY_FILE")
    jwt_public_key_pem: SecretStr = Field(default=SecretStr(""), alias="AI_JWT_PUBLIC_KEY_PEM")
    jwt_issuer: str = Field(default="jj-office-cloudbase", alias="AI_JWT_ISSUER")
    jwt_audience: str = Field(default="jj-office-ai", alias="AI_JWT_AUDIENCE")

    deepseek_api_key: SecretStr = Field(default=SecretStr(""), alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    allowed_models: str = Field(
        default="deepseek-flash,deepseek-v4-pro", alias="AI_ALLOWED_MODELS"
    )
    default_model: str = Field(default="deepseek-flash", alias="AI_DEFAULT_MODEL")
    max_output_tokens: int = Field(default=32768, ge=1, le=393216, alias="AI_MAX_OUTPUT_TOKENS")
    max_request_bytes: int = Field(
        default=4 * 1024 * 1024, ge=1024, le=32 * 1024 * 1024, alias="AI_MAX_REQUEST_BYTES"
    )
    connect_timeout_seconds: float = Field(
        default=10.0, gt=0, le=120, alias="AI_CONNECT_TIMEOUT_SECONDS"
    )
    request_timeout_seconds: float = Field(
        default=660.0, gt=0, le=1200, alias="AI_REQUEST_TIMEOUT_SECONDS"
    )
    upstream_initial_retries: int = Field(
        default=1, ge=0, le=3, alias="AI_UPSTREAM_INITIAL_RETRIES"
    )

    redis_url: SecretStr = Field(default=SecretStr(""), alias="AI_REDIS_URL")
    rate_limit_requests_per_minute: int = Field(
        default=30, ge=1, le=10000, alias="AI_RATE_LIMIT_REQUESTS_PER_MINUTE"
    )
    max_concurrent_requests_per_user: int = Field(
        default=3, ge=1, le=100, alias="AI_MAX_CONCURRENT_REQUESTS_PER_USER"
    )

    quota_service_url: str = Field(default="", alias="AI_QUOTA_SERVICE_URL")
    quota_service_token: SecretStr = Field(default=SecretStr(""), alias="AI_QUOTA_SERVICE_TOKEN")

    @field_validator("deepseek_base_url", "quota_service_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        return value.strip().rstrip("/")

    @model_validator(mode="after")
    def validate_security(self) -> Settings:
        if self.environment == "production" and self.auth_mode != "jwt":
            raise ValueError("AI_AUTH_MODE=jwt is required in production")
        if self.auth_mode == "static" and not self.parsed_static_tokens:
            raise ValueError("AI_STATIC_TOKENS must contain at least one token in static mode")
        if self.auth_mode == "jwt" and not self.jwt_public_key:
            raise ValueError("A JWT public key is required in jwt mode")
        if self.default_model not in self.parsed_allowed_models:
            raise ValueError("AI_DEFAULT_MODEL must be included in AI_ALLOWED_MODELS")
        if bool(self.quota_service_url) != bool(self.quota_service_token.get_secret_value()):
            raise ValueError("AI_QUOTA_SERVICE_URL and AI_QUOTA_SERVICE_TOKEN must be set together")
        return self

    @property
    def parsed_allowed_models(self) -> tuple[str, ...]:
        models = (item.strip() for item in self.allowed_models.split(","))
        return tuple(dict.fromkeys(item for item in models if item))

    @property
    def parsed_static_tokens(self) -> tuple[str, ...]:
        return tuple(item.strip() for item in self.static_tokens.split(",") if item.strip())

    @property
    def parsed_cors_allowed_origins(self) -> tuple[str, ...]:
        origins = tuple(
            item.strip() for item in self.cors_allowed_origins.split(",") if item.strip()
        )
        return origins or ("*",)

    @property
    def jwt_public_key(self) -> str:
        inline = self.jwt_public_key_pem.get_secret_value().strip()
        if inline:
            return inline.replace("\\n", "\n")
        if self.jwt_public_key_file:
            path = Path(self.jwt_public_key_file).expanduser()
            if path.is_file():
                return path.read_text(encoding="utf-8")
        return ""

    @property
    def ready(self) -> bool:
        return bool(self.deepseek_api_key.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    return Settings()
