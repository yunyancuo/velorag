from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

from velorag.core.settings import get_settings

Base = declarative_base()

_engine = None
_SessionLocal = None


def _ensure_sqlite_parent_dir(db_url: str) -> None:
    """为文件型 SQLite 数据库自动创建父目录。"""
    if not db_url.startswith("sqlite:///"):
        return
    raw_path = db_url.removeprefix("sqlite:///")
    if not raw_path or raw_path == ":memory:":
        return
    Path(raw_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        db_url = settings.db.url

        if "sqlite" in db_url:
            _ensure_sqlite_parent_dir(db_url)
            engine_kwargs: dict = {
                "connect_args": {"check_same_thread": False},
                "echo": settings.db.echo,
            }
            if ":memory:" in db_url:
                engine_kwargs["poolclass"] = StaticPool

            _engine = create_engine(db_url, **engine_kwargs)

            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        else:
            _engine = create_engine(
                db_url,
                pool_size=settings.db.pool_size,
                max_overflow=settings.db.max_overflow,
                pool_pre_ping=True,
                echo=settings.db.echo,
            )

    return _engine


def reset_engine() -> None:
    """释放当前数据库引擎与会话工厂（用于测试隔离或配置重载）。"""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


def get_session():
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SessionLocal()


def create_all():
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
