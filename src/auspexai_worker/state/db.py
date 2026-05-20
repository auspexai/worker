"""SQLite connection wrapper + migration runner for the worker's local state.

Worker-side state is small (single-row identity in M1; manifest pins, receipts
metadata, account binding in later milestones). We use the same migration
convention as the coordinator: `migrations_sql/NNNN_name.sql` files, applied
in sequence, recorded in a `schema_migrations` table.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

MIGRATION_PATTERN = re.compile(r"^(\d{4,})_(.+)\.sql$")
DEFAULT_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations_sql"


class MigrationError(Exception):
    """Raised on malformed migration filenames or non-sequential versions."""


class Database:
    """Thin wrapper around a sqlite3.Connection with WAL + foreign keys on."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    def close(self) -> None:
        self._conn.close()


class MigrationRunner:
    """Apply pending migrations in `migrations_sql/` to a Database."""

    def __init__(self, db: Database, migrations_dir: Path | None = None) -> None:
        self.db = db
        self.migrations_dir = migrations_dir or DEFAULT_MIGRATIONS_DIR

    def apply_all(self) -> list[int]:
        """Apply pending migrations. Returns the versions that were applied."""
        self._ensure_schema_migrations_table()
        applied = self._already_applied()
        pending = self._discover_pending(applied)
        new_versions: list[int] = []
        for version, name, sql_path in pending:
            self.db.connection.executescript(sql_path.read_text(encoding="utf-8"))
            self.db.connection.execute(
                "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
                (version, name),
            )
            new_versions.append(version)
        return new_versions

    # ---- private --------------------------------------------------------

    def _ensure_schema_migrations_table(self) -> None:
        self.db.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

    def _already_applied(self) -> set[int]:
        rows = self.db.connection.execute("SELECT version FROM schema_migrations").fetchall()
        return {row["version"] for row in rows}

    def _discover_pending(self, applied: set[int]) -> list[tuple[int, str, Path]]:
        if not self.migrations_dir.exists():
            raise MigrationError(f"migrations directory not found: {self.migrations_dir}")
        candidates: list[tuple[int, str, Path]] = []
        for path in sorted(self.migrations_dir.iterdir()):
            if not path.is_file() or not path.name.endswith(".sql"):
                continue
            match = MIGRATION_PATTERN.match(path.name)
            if match is None:
                raise MigrationError(f"malformed migration filename: {path.name}")
            version = int(match.group(1))
            name = match.group(2)
            candidates.append((version, name, path))
        candidates.sort(key=lambda triple: triple[0])
        seen: set[int] = set()
        for version, _, path in candidates:
            if version in seen:
                raise MigrationError(f"duplicate migration version {version} at {path}")
            seen.add(version)
        return [c for c in candidates if c[0] not in applied]
