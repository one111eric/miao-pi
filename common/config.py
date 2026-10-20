import os
from enum import StrEnum

from pydantic import Field
from pydantic_settings import BaseSettings


class LogFormat(StrEnum):
    JSON = "json"
    TEXT = "text"


class SvcSettings(BaseSettings):
    app_name: str = "miao-pi"
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.JSON
    port: int = Field(default=9091, validation_alias="PORT")
