"""Linux backend: bubblewrap with every namespace unshared.

Needs bubblewrap 0.4.0 or later and unprivileged user namespaces (on
Ubuntu 24.04 and later, an AppArmor profile for bwrap or
kernel.apparmor_restrict_unprivileged_userns=0). The startup self-test
catches a setup where it doesn't work. See docs/sandbox.md.
"""

import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from app.agent.sandbox.base import CommandSettings, ConfinedSandbox, RunDir
from app.agent.sandbox.policy import SandboxConfigError, SandboxPolicy

# Mounted read-only when they exist.
SYSTEM_READ_PATHS = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/etc")


def _read_only(path: Path) -> list[str]:
    if path.is_symlink():
        # On merged-/usr systems /bin and friends are links into /usr.
        return ["--symlink", os.readlink(path), str(path)]
    if path.exists():
        return ["--ro-bind", str(path), str(path)]
    return []


class BubblewrapSandbox(ConfinedSandbox):
    name = "bubblewrap"

    def __init__(
        self,
        policy: SandboxPolicy,
        settings: CommandSettings | None = None,
    ) -> None:
        super().__init__(policy, settings)
        executable = shutil.which("bwrap")
        if executable is None:
            raise SandboxConfigError("bwrap isn't installed")
        self.executable = executable

    @staticmethod
    def available() -> bool:
        return sys.platform.startswith("linux") and shutil.which("bwrap") is not None

    def masks(self) -> list[str]:
        """Arguments hiding every denied file and directory in the workspace.

        bubblewrap has no pattern rules, so this walks the workspace before
        each command. It ignores .gitignore (a .env is usually ignored) and
        doesn't descend into denied directories. Symlinks aren't masked: one
        pointing at a denied path reaches the mask, and one pointing outside
        the workspace leads nowhere inside the sandbox.
        """
        workspace = self.policy.workspace
        args: list[str] = []
        for dirpath, dirnames, filenames in os.walk(workspace.root):
            current = Path(dirpath)
            kept: list[str] = []
            for name in sorted(dirnames):
                path = current / name
                if path.is_symlink() or not workspace.is_denied(path):
                    kept.append(name)
                    continue
                args += ["--tmpfs", str(path)]
            dirnames[:] = kept
            for name in sorted(filenames):
                path = current / name
                if not path.is_symlink() and workspace.is_denied(path):
                    args += ["--ro-bind", "/dev/null", str(path)]
        return args

    def confine(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        run: RunDir,
    ) -> list[str]:
        root = str(self.policy.workspace.root)
        scratch = str(run.scratch)
        args = [self.executable, "--unshare-all", "--die-with-parent", "--new-session"]
        for path in (*map(Path, SYSTEM_READ_PATHS), *self.policy.read_paths):
            args += _read_only(path)
        # The workspace and scratch keep their real paths, so tools inside
        # see the same paths as on the host. The PID namespace ends every
        # process when the command does, so there's nothing to sweep.
        args += ["--bind", root, root, "--bind", scratch, scratch]
        args += ["--dev", "/dev", "--proc", "/proc"]
        # After the workspace bind, so the masks sit on top of it.
        args += self.masks()
        return [*args, "--chdir", str(cwd), "--", *argv]
