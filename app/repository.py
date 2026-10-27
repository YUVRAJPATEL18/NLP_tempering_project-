"""Replace this adapter with the host system's repository when integrating."""
import sqlite3
from pathlib import Path
from typing import Protocol


class ReferenceRepository(Protocol):
    def get_photo(self, reference_id: str) -> bytes | None: ...


class SQLiteReferenceRepository:
    def __init__(self, database: Path):
        self.database = database
        database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(database) as connection:
            connection.execute('CREATE TABLE IF NOT EXISTS reference_faces '
                               '(reference_id TEXT PRIMARY KEY, photo BLOB NOT NULL)')

    def get_photo(self, reference_id: str) -> bytes | None:
        with sqlite3.connect(self.database) as connection:
            row = connection.execute('SELECT photo FROM reference_faces WHERE reference_id = ?',
                                     (reference_id,)).fetchone()
        return bytes(row[0]) if row else None

    def save_photo(self, reference_id: str, photo: bytes) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.execute('INSERT INTO reference_faces(reference_id, photo) VALUES (?, ?) '
                               'ON CONFLICT(reference_id) DO UPDATE SET photo=excluded.photo',
                               (reference_id, photo))
