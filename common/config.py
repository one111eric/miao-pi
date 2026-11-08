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


class SandboxBackend(StrEnum):
    AUTO = "auto"
    SEATBELT = "seatbelt"
    BUBBLEWRAP = "bubblewrap"
    # Command tools are disabled; there is no unsandboxed setting.
    OFF = "off"


DEFAULT_SANDBOX_PATH = "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin"


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
    # LangGraph steps per request; each tool round takes two.
    agent_max_steps: int = Field(default=25, ge=5)
    agent_run_timeout: float = Field(default=300, gt=0)
    # A larger last resort after the normal deadline stops between graph steps.
    agent_run_backstop_timeout: float = Field(default=60, gt=0)
    # Serves the test page and the thread history endpoint. No auth until 7b.
    dev_ui_enabled: bool = False
    # The command sandbox (step 5b); see docs/sandbox.md.
    sandbox_backend: SandboxBackend = SandboxBackend.AUTO
    sandbox_command_timeout: float = Field(default=120, gt=0)
    sandbox_max_output_bytes: int = Field(default=50 * 1024, gt=0)
    # Read-only paths for toolchains under the home directory.
    sandbox_read_paths: tuple[str, ...] = ()
    sandbox_path: str = DEFAULT_SANDBOX_PATH
    # Extra variables for every command; the service's own aren't passed on.
    sandbox_env: dict[str, str] = {}

    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )
