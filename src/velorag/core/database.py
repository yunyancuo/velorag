from __future__ import annotations

from sqlalchemy import inspect, text

from velorag.db.base import Base, create_all, get_engine, get_session, reset_engine
from velorag.db.deps import get_db


def _migrate_sqlite_columns() -> None:
    """向后兼容补齐旧版 SQLite 数据库中新增的可选列。"""
    engine = get_engine()
    if engine.dialect.name != "sqlite":
        return

    inspector = inspect(engine)
    migrations: dict[str, list[tuple[str, str]]] = {
        "velorag_knowledge_bases": [
            ("owner_id", "VARCHAR(64)"),
            ("is_public", "BOOLEAN NOT NULL DEFAULT 1"),
            ("chunk_strategy", "VARCHAR(32) NOT NULL DEFAULT 'recursive'"),
            ("retrieval_mode", "VARCHAR(32) NOT NULL DEFAULT 'hybrid'"),
        ],
        "velorag_chat_sessions": [
            ("owner_id", "VARCHAR(64)"),
        ],
    }

    with engine.begin() as conn:
        existing_tables = set(inspector.get_table_names())
        for table_name, columns in migrations.items():
            if table_name not in existing_tables:
                continue
            existing_cols = {col["name"] for col in inspector.get_columns(table_name)}
            for col_name, col_ddl in columns:
                if col_name not in existing_cols:
                    conn.execute(
                        text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_ddl}")
                    )


def init_db():
    """注册全部 ORM 模型、创建缺失表结构并兼容补齐新增字段。"""
    import velorag.chat.models  # noqa: F401
    import velorag.db.models  # noqa: F401
    import velorag.knowledge.models  # noqa: F401

    create_all()
    _migrate_sqlite_columns()


__all__ = ["Base", "get_db", "get_engine", "get_session", "init_db", "reset_engine"]
