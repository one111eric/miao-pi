import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

from pythonjsonlogger.json import JsonFormatter

from common.config import LogFormat

_logging_request_context: ContextVar[dict[str, object]] = ContextVar(
    "logging_context", default={}
)

LOGGING_FORMAT = (
    "%(asctime)s.%(msecs)03dZ level=%(levelname)s %(context)s"
    "name=%(name)s module=%(module)s line=%(lineno)d op=%(funcName)s msg=%(message)s"
)

LOGGING_DATEFMT = "%Y-%m-%dT%H:%M:%S"

_DEFAULT_RECORD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
)


def reset_context(state: Token[dict[str, object]]) -> None:
    """Resets the state of the logging context to that at the given state."""
    _logging_request_context.reset(state)


def update_context(**kwargs: object) -> Token[dict[str, object]]:
    """Updates the state of the logging context.
    Returns a token that can be used to recover the previous state using `reset_context`
    """
    context = {**_logging_request_context.get(), **kwargs}
    return _logging_request_context.set(context)


@contextmanager
def logging_context(**kwargs: object) -> Iterator[None]:
    token = update_context(**kwargs)
    try:
        yield
    finally:
        reset_context(token)


def _get_context() -> dict[str, object]:
    return _logging_request_context.get()


class RequestFormatter(logging.Formatter):
    """Text-based structured formatter (key=value output)."""

    def format(self, record: logging.LogRecord) -> str:
        contexts: list[str] = []

        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _DEFAULT_RECORD_ATTRS and k != "context"
        }
        contexts.extend(f"{k}={v} " for k, v in extras.items())

        ctx = _get_context()
        contexts.extend(f"{k}={v} " for k, v in ctx.items())

        exc_info = record.exc_info
        if exc_info and exc_info[0] is not None:
            contexts.append(f"exception_type={exc_info[0].__name__} ")

        record.context = "".join(contexts)
        return super().format(record)


class JsonRequestFormatter(JsonFormatter):
    """JSON-lines formatter that merges extra fields and request context."""

    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s %(msecs)03d %(levelname)s %(name)s %(message)s %(context)s",
            rename_fields={"levelname": "level"},
            datefmt=LOGGING_DATEFMT,
        )

    def add_fields(
        self,
        log_data: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_data, record, message_dict)
        msecs = str(int(log_data.get("msecs", 0))).zfill(3)
        log_data["timestamp"] = f"{log_data.get('asctime', '')}.{msecs}"
        log_data.pop("asctime", None)
        log_data.pop("msecs", None)

    def format(self, record: logging.LogRecord) -> str:
        record.context = _get_context()
        return super().format(record)


def configure_logger(log_level: str, log_format: LogFormat = LogFormat.JSON) -> None:
    handler = logging.StreamHandler()

    if log_format == LogFormat.JSON:
        formatter: logging.Formatter = JsonRequestFormatter()
    else:
        formatter = RequestFormatter(LOGGING_FORMAT, datefmt=LOGGING_DATEFMT)

    handler.setFormatter(formatter)
    level = getattr(logging, log_level.upper())

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)
