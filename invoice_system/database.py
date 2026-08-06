from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Final


DEFAULT_DATABASE_PATH: Final = Path("inventory.db")

SEED_INVENTORY: Final[tuple[tuple[str, int], ...]] = (
    ("WidgetA", 15),
    ("WidgetB", 10),
    ("GadgetX", 5),
    ("FakeItem", 0),
)


def connect(database_path: str | Path = DEFAULT_DATABASE_PATH) -> sqlite3.Connection:
    """
    Open a SQLite connection with dictionary-like rows.
    """

    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> Path:
    """
    Create and seed the local inventory database.

    The operation is idempotent and can safely be run multiple times.
    """

    path = Path(database_path)

    with connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS inventory (
                item TEXT PRIMARY KEY,
                stock INTEGER NOT NULL
            )
            """
        )

        connection.executemany(
            """
            INSERT INTO inventory (item, stock)
            VALUES (?, ?)
            ON CONFLICT(item) DO UPDATE SET
                stock = excluded.stock
            """,
            SEED_INVENTORY,
        )

        connection.commit()

    return path


def lookup_inventory(
    item_name: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, str | int] | None:
    """
    Find one inventory item using normalized, case-insensitive matching.

    This handles harmless formatting differences such as:
    - "widgeta", "WidgetA", "WIDGETA"

    It intentionally does not perform fuzzy matching. Automatically mapping a
    typo to a financial inventory record could validate the wrong product.
    """

    normalized_name = item_name.strip()

    if not normalized_name:
        return None

    with connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT item, stock
            FROM inventory
            WHERE LOWER(TRIM(item)) = LOWER(TRIM(?))
            LIMIT 1
            """,
            (normalized_name,),
        ).fetchone()

    if row is None:
        return None

    return {
        "item": str(row["item"]),
        "stock": int(row["stock"]),
    }


if __name__ == "__main__":
    created_path = initialize_database()
    print(f"Inventory database initialized at {created_path.resolve()}")