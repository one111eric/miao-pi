from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, RedirectResponse

from app.ui.gate import require_dev_ui

STATIC_DIR = Path(__file__).parent / "static"

# Scripts and styles are separate files, so nothing inline is ever allowed;
# a reply that smuggles in HTML can't run script even if it were rendered.
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'self'; style-src 'self'; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)

_HEADERS = {
    # Edits to the page show up on reload while developing.
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
}

router = APIRouter(
    prefix="/ui",
    dependencies=[Depends(require_dev_ui)],
    include_in_schema=False,
)


@router.get("")
async def redirect_to_page() -> RedirectResponse:
    # The page lives at /ui/ so its relative asset links resolve under /ui.
    return RedirectResponse("/ui/")


@router.get("/")
async def page() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        media_type="text/html",
        headers={**_HEADERS, "Content-Security-Policy": CONTENT_SECURITY_POLICY},
    )


@router.get("/app.js")
async def script() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "app.js",
        media_type="text/javascript",
        headers=_HEADERS,
    )


@router.get("/app.css")
async def stylesheet() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "app.css",
        media_type="text/css",
        headers=_HEADERS,
    )


@router.get("/sse.js")
async def stream_parser() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "sse.js",
        media_type="text/javascript",
        headers=_HEADERS,
    )
