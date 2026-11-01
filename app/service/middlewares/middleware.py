import logging
import re
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from common.logging_helper import logging_context

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

# Supplied ids end up in log lines, so anything else is replaced with a new id.
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")


def _request_id_from(scope: Scope) -> str:
    supplied = Headers(scope=scope).get(REQUEST_ID_HEADER)
    if supplied is not None and _VALID_REQUEST_ID.fullmatch(supplied):
        return supplied
    return str(uuid4())


class RequestContextMiddleware:
    """Attach a request id to logs and return it in the response headers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id_from(scope)
        response_started = False

        async def send_with_request_id(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                response_headers = MutableHeaders(scope=message)
                response_headers[REQUEST_ID_HEADER] = request_id
            await send(message)

        with logging_context(request_id=request_id):
            try:
                await self.app(scope, receive, send_with_request_id)
            except Exception:
                # Handled here so the error is logged with the request id and
                # the 500 response still carries the header.
                logger.exception("Unhandled error while processing request")
                if response_started:
                    raise
                response = JSONResponse(
                    status_code=500,
                    content={"detail": "Internal server error"},
                )
                await response(scope, receive, send_with_request_id)
