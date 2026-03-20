from __future__ import annotations

import sqlite3

from duo_rag.schema import MetadataField, MetadataSchema
from duo_rag.stores.base import RelationalStore

REVERSE_TYPE_MAP = {"TEXT": "text", "INTEGER": "integer"}
TYPE_MAP = {"text": "TEXT", "integer": "INTEGER"}

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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS _schema_fields (
                    name TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS _document_hashes (
                    doc_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL
                )
                """
            )
            for field in schema.fields:
                conn.execute(
                    "INSERT OR REPLACE INTO _schema_fields (name, type, description) VALUES (?, ?, ?)",
                    (field.name, field.type, field.description),
                )
            conn.commit()
        finally:
            conn.close()

    def insert(self, doc_id: str, metadata: dict) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            existing = conn.execute(
                f"SELECT id FROM {MetadataSchema.TABLE_NAME} WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if existing:
                for key, value in metadata.items():
                    if value is not None:
                        conn.execute(
                            f"UPDATE {MetadataSchema.TABLE_NAME} SET {key} = COALESCE({key}, ?) WHERE doc_id = ?",
                            (value, doc_id),
                        )
            else:
                columns = ["doc_id"] + list(metadata.keys())
                values = [doc_id] + list(metadata.values())
                placeholders = ", ".join("?" for _ in columns)
                column_names = ", ".join(columns)
                conn.execute(
                    f"INSERT INTO {MetadataSchema.TABLE_NAME} ({column_names}) VALUES ({placeholders})",
                    values,
                )
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
            # Try reading from _schema_fields first (preserves descriptions)
            try:
                cursor = conn.execute(
                    "SELECT name, type, description FROM _schema_fields ORDER BY rowid"
                )
                rows = cursor.fetchall()
                if rows:
                    fields = [
                        MetadataField(name=r[0], type=r[1], description=r[2])
                        for r in rows
                    ]
                    return MetadataSchema(fields=fields)
            except sqlite3.OperationalError:
                pass

            # Fall back to PRAGMA (backwards compat, no descriptions)
            try:
                cursor = conn.execute(
                    f"PRAGMA table_info({MetadataSchema.TABLE_NAME})"
                )
                rows = cursor.fetchall()
            except sqlite3.OperationalError:
                return None

            if not rows:
                return None

            skip_columns = {"id", "doc_id"}
            fields: list[MetadataField] = []
            for row in rows:
                col_name = row[1]
                col_type = row[2]
                if col_name in skip_columns:
                    continue
                mapped_type = REVERSE_TYPE_MAP.get(col_type.upper(), "text")
                fields.append(MetadataField(name=col_name, type=mapped_type, description=""))

            return MetadataSchema(fields=fields)
        finally:
            conn.close()

    def add_column(self, field: MetadataField) -> None:
        sql_type = TYPE_MAP[field.type]
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                f"ALTER TABLE {MetadataSchema.TABLE_NAME} ADD COLUMN {field.name} {sql_type}"
            )
            conn.execute(
                "INSERT OR REPLACE INTO _schema_fields (name, type, description) VALUES (?, ?, ?)",
                (field.name, field.type, field.description),
            )
            conn.commit()
        finally:
            conn.close()

    def drop_column(self, field_name: str) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                f"ALTER TABLE {MetadataSchema.TABLE_NAME} DROP COLUMN {field_name}"
            )
            conn.execute(
                "DELETE FROM _schema_fields WHERE name = ?",
                (field_name,),
            )
            conn.commit()
        finally:
            conn.close()

    def get_empty_fields(self) -> list[str]:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(f"PRAGMA table_info({MetadataSchema.TABLE_NAME})")
            pragma_rows = cursor.fetchall()

            skip = {"id", "doc_id"}
            empty_fields = []
            for row in pragma_rows:
                col_name = row[1]
                if col_name in skip:
                    continue
                cursor = conn.execute(
                    f"SELECT COUNT(*) FROM {MetadataSchema.TABLE_NAME} WHERE {col_name} IS NOT NULL"
                )
                count = cursor.fetchone()[0]
                if count == 0:
                    empty_fields.append(col_name)
            return empty_fields
        finally:
            conn.close()

    def get_unpopulated_fields(self, schema: MetadataSchema) -> list[MetadataField]:
        empty = set(self.get_empty_fields())
        return [f for f in schema.fields if f.name in empty]

    def update_metadata_field(self, doc_id: str, field_name: str, value: object) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                f"UPDATE {MetadataSchema.TABLE_NAME} SET {field_name} = ? WHERE doc_id = ?",
                (value, doc_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_document_hash(self, doc_id: str) -> str | None:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT content_hash FROM _document_hashes WHERE doc_id = ?",
                (doc_id,),
            )
            row = cursor.fetchone()
            return row[0] if row else None
        except sqlite3.OperationalError:
            return None
        finally:
            conn.close()

    def set_document_hash(self, doc_id: str, content_hash: str) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO _document_hashes (doc_id, content_hash) VALUES (?, ?)",
                (doc_id, content_hash),
            )
            conn.commit()
        finally:
            conn.close()

    def delete_document(self, doc_id: str) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                f"DELETE FROM {MetadataSchema.TABLE_NAME} WHERE doc_id = ?",
                (doc_id,),
            )
            conn.execute(
                "DELETE FROM _document_hashes WHERE doc_id = ?",
                (doc_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def clear_all_data(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(f"DELETE FROM {MetadataSchema.TABLE_NAME}")
            conn.execute("DELETE FROM _document_hashes")
            conn.commit()
        finally:
            conn.close()
