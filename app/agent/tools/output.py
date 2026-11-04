"""Output limits and truncation shared by the read-only tools.

The limits follow pi's coding agent: whichever of the line or byte cap is hit
first wins, lines are never cut in half, and every cut ends with a notice that
tells the model how to get the rest.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal, TextIO

MAX_LINES = 2000
MAX_BYTES = 50 * 1024
# grep shows at most this much of each line.
MAX_LINE_CHARS = 500
# Files whose first block contains a NUL byte are treated as binary.
BINARY_SNIFF_BYTES = 8192


def format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / (1024 * 1024):.1f}MB"


@dataclass(frozen=True)
class Head:
    """The leading lines that fit, and which cap stopped the rest."""

    lines: list[str]
    truncated_by: Literal["lines", "bytes"] | None
    # The first line alone was over the byte cap, so nothing could be shown.
    first_line_too_long: bool


def take_head(
    lines: Iterable[str],
    *,
    max_lines: int = MAX_LINES,
    max_bytes: int = MAX_BYTES,
) -> Head:
    """Keep whole lines from the start until either cap is reached.

    When a cap is hit, one line past the kept ones has been consumed from
    ``lines``.
    """
    kept: list[str] = []
    size = 0
    for line in lines:
        if len(kept) >= max_lines:
            return Head(kept, "lines", first_line_too_long=False)
        # Every line after the first adds its separating newline.
        line_size = len(line.encode()) + (1 if kept else 0)
        if size + line_size > max_bytes:
            return Head(kept, "bytes", first_line_too_long=not kept)
        kept.append(line)
        size += line_size
    return Head(kept, None, first_line_too_long=False)


def truncate_line(line: str) -> tuple[str, bool]:
    if len(line) <= MAX_LINE_CHARS:
        return line, False
    return f"{line[:MAX_LINE_CHARS]}... [truncated]", True


def iter_lines(stream: TextIO, *, max_chars: int) -> Iterator[str]:
    """Yield lines without their newline, each cut to at most ``max_chars``.

    The rest of an over-long line is read and dropped, so memory stays bounded
    even for a huge file with no newlines.
    """
    while chunk := stream.readline(max_chars):
        if chunk.endswith("\n"):
            yield chunk[:-1]
            continue
        if len(chunk) == max_chars:
            while (rest := stream.readline(max_chars)) and not rest.endswith("\n"):
                pass
        yield chunk


def with_notices(text: str, notices: list[str]) -> str:
    if not notices:
        return text
    return f"{text}\n\n[{'. '.join(notices)}]"
