import logging

from common.logging_helper import RequestFormatter, logging_context


def test_nested_logging_context_overrides_existing_key() -> None:
    formatter = RequestFormatter("%(context)s%(message)s")
    record = logging.LogRecord("test", logging.INFO, "", 0, "message", (), None)

    with logging_context(request_id="outer"):
        with logging_context(request_id="inner", details=["value"]):
            rendered = formatter.format(record)

    assert "request_id=inner" in rendered
    assert "request_id=outer" not in rendered
    assert "details=['value']" in rendered
