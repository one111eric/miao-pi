from fastapi import HTTPException, Request, status


def require_dev_ui(request: Request) -> None:
    """Hide developer-only routes unless ``dev_ui_enabled`` is set.

    The 404 matches FastAPI's own, so a service with the UI off looks like it
    has no such routes.
    """
    if not getattr(request.app.state, "dev_ui_enabled", False):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
