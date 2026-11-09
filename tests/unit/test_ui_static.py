import re

import pytest

from app.ui.router import STATIC_DIR

# Replies and tool results are untrusted text, so the page must never parse
# them as HTML or code.
FORBIDDEN = (
    "innerHTML",
    "outerHTML",
    "insertAdjacentHTML",
    "document.write",
    "eval(",
    "new Function",
)


@pytest.mark.parametrize("name", ["index.html", "app.js", "sse.js", "app.css"])
def test_the_page_never_turns_text_into_html_or_code(name: str) -> None:
    content = (STATIC_DIR / name).read_text(encoding="utf-8")

    for api in FORBIDDEN:
        assert api not in content, f"{name} uses {api}"


def test_the_page_has_no_inline_script_or_style() -> None:
    # The CSP blocks inline code, so it would silently break the page.
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert "<script>" not in page
    assert "<style" not in page
    assert not re.search(r"\sstyle\s*=", page)
    # Event handler attributes such as onclick= are inline script too.
    assert not re.search(r"\son[a-z]+\s*=", page)
