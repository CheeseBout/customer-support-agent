"""Per-session workspace: small files the agent produced for a conversation (SPEC 12.3).

Layout: `<root>/<hash of user id>/<session id>/<artifact>`. The user hash is part of the path,
so a session id alone can never address another user's files, and only a fixed list of artifact
names is accepted, so a name can never escape the folder.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from support_agent.core.principal import session_namespace

ARTIFACT_NAMES = ("comparison_table.md", "request_summary.md")
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class WorkspaceError(ValueError):
    """An unknown artifact name or a malformed session id."""


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _session_dir(self, user_id: str, session_id: str) -> Path:
        if not SESSION_ID.match(session_id):
            raise WorkspaceError("invalid session id")
        return self.root / session_namespace(user_id) / session_id

    @staticmethod
    def _check_name(name: str) -> str:
        if name not in ARTIFACT_NAMES:
            raise WorkspaceError(f"unknown artifact {name!r}")
        return name

    def write(self, user_id: str, session_id: str, name: str, content: str) -> None:
        folder = self._session_dir(user_id, session_id)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / self._check_name(name)).write_text(content, encoding="utf-8")

    def read(self, user_id: str, session_id: str, name: str) -> str | None:
        path = self._session_dir(user_id, session_id) / self._check_name(name)
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def names(self, user_id: str, session_id: str) -> list[str]:
        folder = self._session_dir(user_id, session_id)
        return [n for n in ARTIFACT_NAMES if (folder / n).is_file()]

    def delete_session(self, user_id: str, session_id: str) -> None:
        shutil.rmtree(self._session_dir(user_id, session_id), ignore_errors=True)

    def purge_older_than(self, days: int) -> int:
        """Delete session folders untouched for `days` days. Returns how many were removed."""
        if not self.root.is_dir():
            return 0
        cutoff = time.time() - days * 86400
        removed = 0
        for user_dir in self.root.iterdir():
            if not user_dir.is_dir():
                continue
            for session_dir in user_dir.iterdir():
                if session_dir.is_dir() and session_dir.stat().st_mtime < cutoff:
                    shutil.rmtree(session_dir, ignore_errors=True)
                    removed += 1
        return removed
