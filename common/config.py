from enum import StrEnum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogFormat(StrEnum):
    JSON = "json"
    TEXT = "text"


class LLMProvider(StrEnum):
    OPENAI = "openai"
    LOCAL = "local"


class SvcSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "miao-pi"
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.JSON
    port: int = Field(default=9091, validation_alias="PORT")
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
