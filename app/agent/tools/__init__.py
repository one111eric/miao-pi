"""Read-only workspace tools, shaped after pi's coding agent.

pi's `read`, `ls`, `find`, and `grep` keep their names, parameters, limits,
and truncation notices. The differences: every path goes through `Workspace`
(denied and escaping paths are rejected or hidden), results are relative to
the project root, `find` and `grep` run in Python so the deny list and
`.gitignore` apply to them, and binary files are skipped.
"""

from langchain_core.tools import BaseTool

from app.agent.tools.find import make_find_tool
from app.agent.tools.grep import make_grep_tool
from app.agent.tools.ls import make_ls_tool
from app.agent.tools.read import make_read_tool
from app.agent.workspace import Workspace


def make_read_tools(workspace: Workspace) -> list[BaseTool]:
    """Create the read-only tools, all bound to one workspace."""
    return [
        make_read_tool(workspace),
        make_ls_tool(workspace),
        make_find_tool(workspace),
        make_grep_tool(workspace),
    ]
