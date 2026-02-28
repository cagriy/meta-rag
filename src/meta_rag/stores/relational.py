from __future__ import annotations

import sqlite3

from meta_rag.schema import MetadataField, MetadataSchema
from meta_rag.stores.base import RelationalStore

REVERSE_TYPE_MAP = {"TEXT": "text", "INTEGER": "integer"}

BLACKLISTED_KEYWORDS = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "EXEC"}


class SQLiteRelationalStore(RelationalStore):
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.schema: MetadataSchema | None = None

    def initialize(self, schema: MetadataSchema) -> None:
        self.schema = schema
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(schema.to_ddl())
            conn.commit()
        finally:
            conn.close()

    def insert(self, doc_id: str, chunk_id: str, metadata: dict) -> None:
        columns = ["doc_id", "chunk_id"] + list(metadata.keys())
        values = [doc_id, chunk_id] + list(metadata.values())
        placeholders = ", ".join("?" for _ in columns)
        column_names = ", ".join(columns)
        sql = f"INSERT INTO {MetadataSchema.TABLE_NAME} ({column_names}) VALUES ({placeholders})"

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(sql, values)
            conn.commit()
        finally:
            conn.close()

    def execute_sql(self, sql: str) -> list[dict]:
        normalized = sql.strip().upper()

        if not normalized.startswith("SELECT"):
            raise ValueError("Only SELECT statements are allowed.")

        for keyword in BLACKLISTED_KEYWORDS:
            if keyword in normalized:
                raise ValueError(f"Forbidden SQL keyword detected: {keyword}")

        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_schema_from_db(self) -> MetadataSchema | None:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                f"PRAGMA table_info({MetadataSchema.TABLE_NAME})"
            )
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            return None
        finally:
            conn.close()

        if not rows:
            return None

        skip_columns = {"id", "doc_id", "chunk_id"}
        fields: list[MetadataField] = []

        for row in rows:
            # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
            col_name = row[1]
            col_type = row[2]

            if col_name in skip_columns:
                continue

            mapped_type = REVERSE_TYPE_MAP.get(col_type.upper(), "text")
            fields.append(MetadataField(name=col_name, type=mapped_type, description=""))

        return MetadataSchema(fields=fields)
