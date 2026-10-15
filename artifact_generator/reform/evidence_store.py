"""Read the frozen evidence SQLite files without the WorldReasoner runtime."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (
        parsed
        if parsed.tzinfo is not None
        else parsed.replace(tzinfo=timezone.utc)
    )


def _direct_evidence_pack(
    db_path: Path,
    question_id: str,
    cutoff: datetime,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return the original runner's cutoff-safe, newest-first evidence pack."""
    normalized_cutoff = (
        cutoff
        if cutoff.tzinfo is not None
        else cutoff.replace(tzinfo=timezone.utc)
    )
    with sqlite3.connect(str(db_path)) as connection:
        rows = connection.execute(
            """
            SELECT id, title, source, published_date, content
            FROM articles
            WHERE collected_for_question_id = ?
            """,
            (question_id,),
        ).fetchall()
    parsed_rows = [
        (row, _parse_datetime(str(row[3])))
        for row in rows
        if row[3] and _parse_datetime(str(row[3])) < normalized_cutoff
    ]
    parsed_rows.sort(key=lambda item: item[1], reverse=True)
    return [
        {
            "id": str(row[0]),
            "title": str(row[1]),
            "source": str(row[2]),
            "published_date": published_date.isoformat(),
            "excerpt": " ".join(str(row[4] or "").split())[:500],
        }
        for row, published_date in parsed_rows[:limit]
    ]
