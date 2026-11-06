"""macOS backend: a generated Seatbelt profile, applied with sandbox-exec.

``sandbox-exec`` is marked deprecated but ships with current macOS; Claude
Code and Anthropic's sandbox-runtime use it the same way. See
docs/sandbox.md for what each rule is for and how it was checked.
"""

import sys
from collections.abc import Sequence
from pathlib import Path

from app.agent.sandbox.base import CommandSettings, ConfinedSandbox
from app.agent.sandbox.policy import SandboxConfigError, SandboxPolicy

SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")

# Real paths: on macOS, /etc, /var, and /tmp live under /private.
SYSTEM_READ_SUBPATHS = (
    "/usr",
    "/bin",
    "/sbin",
    "/System",
    "/Library",
    "/private/etc",
    "/private/var/db",
    "/dev",
    "/opt/homebrew",
    "/Applications/Xcode.app",
)
# Directories tools stat or read while walking a path to its target.
SYSTEM_READ_LITERALS = ("/", "/private", "/etc", "/var", "/tmp")

# Deny everything, then allow only what commands need. Left out on purpose:
# network (no rule, so all of it is denied), mach-lookup (system services,
# some of which launch apps outside the sandbox), and signals to processes
# outside the sandbox. ipc-posix-sem is what Python's multiprocessing needs.
_PROFILE_HEAD = """\
(version 1)
(deny default)
(allow process-exec process-fork sysctl-read)
(allow signal (target same-sandbox))
(allow ipc-posix-sem)
(allow file-read-metadata)
"""

_REGEX_SPECIAL = frozenset(".^$*+?()[]{}|\\")
# Characters allowed inside a [...] class in a deny pattern; anything else
# would need escaping rules the profile language doesn't document.
_CLASS_SAFE = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._")


def _escape(text: str) -> str:
    return "".join(f"\\{char}" if char in _REGEX_SPECIAL else char for char in text)


def _any_case(char: str) -> str:
    upper = char.upper()
    if char != upper and len(upper) == 1:
        return f"[{char}{upper}]"
    return _escape(char)


def _class(body: str, pattern: str) -> str:
    negate = body.startswith("!")
    if negate:
        body = body[1:]
    parts: list[str] = []
    index = 0
    while index < len(body):
        if index + 2 < len(body) and body[index + 1] == "-":
            low, high = body[index], body[index + 2]
            for char in (low, high):
                if char not in _CLASS_SAFE:
                    raise SandboxConfigError(f"unsupported deny pattern: {pattern}")
            parts.append(f"{low}-{high}")
            if low.isalpha() and high.isalpha():
                parts.append(f"{low.upper()}-{high.upper()}")
            index += 3
            continue
        char = body[index]
        if char not in _CLASS_SAFE and char != "-":
            raise SandboxConfigError(f"unsupported deny pattern: {pattern}")
        parts.append(char)
        if char.isalpha():
            parts.append(char.upper())
        index += 1
    # A negated class must not match "/", or it could span path components.
    return f"[^/{''.join(parts)}]" if negate else f"[{''.join(parts)}]"


def glob_to_regex(pattern: str, *, crosses_slash: bool) -> str:
    """Translate a folded deny pattern into a case-insensitive regex.

    Mirrors ``fnmatch``: ``*`` and ``?`` match within one name, or across
    "/" when ``crosses_slash`` (rooted patterns, as ``Workspace`` matches
    them). The profile language has no case flag, so every letter becomes
    a ``[xX]`` class.
    """
    out: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            out.append(".*" if crosses_slash else "[^/]*")
        elif char == "?":
            out.append("." if crosses_slash else "[^/]")
        elif char == "[" and (end := pattern.find("]", index + 2)) != -1:
            out.append(_class(pattern[index + 1 : end], pattern))
            index = end
        else:
            out.append(_any_case(char))
        index += 1
    return "".join(out)


def deny_regexes(policy: SandboxPolicy) -> list[str]:
    """One regex per deny pattern, anchored at the workspace root.

    A pattern without "/" matches a name at any depth; one with "/" is
    rooted at the workspace. Either covers everything beneath a match.
    """
    root = _escape(str(policy.workspace.root))
    regexes = []
    for pattern in policy.workspace.denied_patterns:
        if "/" in pattern:
            body = glob_to_regex(pattern, crosses_slash=True)
            regexes.append(f"^{root}/{body}(/.*)?$")
        else:
            body = glob_to_regex(pattern, crosses_slash=False)
            regexes.append(f"^{root}/(.*/)?{body}(/.*)?$")
    return regexes


def build_profile(policy: SandboxPolicy) -> str:
    """The profile text; paths that vary per run come in as parameters."""
    reads = [f'(subpath "{path}")' for path in SYSTEM_READ_SUBPATHS]
    reads += [f'(literal "{path}")' for path in SYSTEM_READ_LITERALS]
    reads += [
        f'(subpath (param "READ_{index}"))' for index in range(len(policy.read_paths))
    ]
    denies = " ".join(f'(regex #"{regex}")' for regex in deny_regexes(policy))
    return (
        _PROFILE_HEAD
        + f"(allow file-read* {' '.join(reads)})\n"
        + '(allow file-read* file-write* (subpath (param "WORKSPACE"))'
        + ' (subpath (param "SCRATCH")))\n'
        + '(allow file-write* (literal "/dev/null"))\n'
        # Last on purpose: in Seatbelt the last matching rule wins, so the
        # deny list must follow the workspace allow.
        + f"(deny file-read* file-write* {denies})\n"
    )


class SeatbeltSandbox(ConfinedSandbox):
    name = "seatbelt"

    def __init__(
        self,
        policy: SandboxPolicy,
        settings: CommandSettings | None = None,
    ) -> None:
        super().__init__(policy, settings)
        # The root is spliced into the deny regexes, inside a quoted string.
        if '"' in str(policy.workspace.root):
            raise SandboxConfigError('workspace root contains a quote (")')
        self.profile = build_profile(policy)

    @staticmethod
    def available() -> bool:
        return sys.platform == "darwin" and SANDBOX_EXEC.exists()

    def confine(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        scratch: Path,
    ) -> list[str]:
        params = [
            "-D",
            f"WORKSPACE={self.policy.workspace.root}",
            "-D",
            f"SCRATCH={scratch}",
        ]
        for index, path in enumerate(self.policy.read_paths):
            params += ["-D", f"READ_{index}={path}"]
        return [str(SANDBOX_EXEC), "-p", self.profile, *params, *argv]
