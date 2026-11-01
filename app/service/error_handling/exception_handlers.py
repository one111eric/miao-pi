import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from openai import APIError, APITimeoutError

logger = logging.getLogger(__name__)


async def _handle_openai_timeout(
    _: Request,
    exception: Exception,
) -> JSONResponse:
    """Map provider timeouts without exposing provider response details."""
    logger.warning(
        "LLM provider request timed out",
        extra={"provider_error_type": type(exception).__name__},
    )
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={"detail": "LLM provider request timed out"},
    )


async def _handle_openai_error(
    _: Request,
    exception: Exception,
) -> JSONResponse:
    """Map provider failures without exposing provider response details."""
    logger.warning(
        "LLM provider request failed",
        extra={"provider_error_type": type(exception).__name__},
    )
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": "LLM provider request failed"},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register provider exception mappings on the application."""
    app.add_exception_handler(APITimeoutError, _handle_openai_timeout)
    app.add_exception_handler(APIError, _handle_openai_error)
