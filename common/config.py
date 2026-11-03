from enum import StrEnum
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogFormat(StrEnum):
    JSON = "json"
    TEXT = "text"


class LLMProvider(StrEnum):
    OPENAI = "openai"
    LOCAL = "local"


class SvcSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.JSON
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "gpt-4o-mini"
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_timeout: float = Field(default=60, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)
    thread_ttl: float = Field(default=3600, gt=0)
    max_threads: int = Field(default=1000, gt=0)
    workspace_root: Path = Path(".")
    # Added to the patterns `Workspace` always denies.
    workspace_denied_paths: tuple[str, ...] = ()
    workspace_context_max_chars: int = Field(default=20_000, gt=0)

    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )
