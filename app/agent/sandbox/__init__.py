"""Running commands so they can't reach beyond the workspace (step 5b).

See docs/sandbox.md for the guarantees, the decisions, and how each backend
was checked.
"""

from app.agent.sandbox.base import CommandSettings, ConfinedSandbox, RunDir, Sandbox
from app.agent.sandbox.policy import (
    SandboxConfigError,
    SandboxPolicy,
    resolve_read_paths,
)
from app.agent.sandbox.runner import CommandResult, ContainmentError

__all__ = [
    "CommandResult",
    "CommandSettings",
    "ConfinedSandbox",
    "ContainmentError",
    "RunDir",
    "Sandbox",
    "SandboxConfigError",
    "SandboxPolicy",
    "resolve_read_paths",
]
