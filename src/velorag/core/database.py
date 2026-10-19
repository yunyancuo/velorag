from __future__ import annotations

from velorag.db.base import Base, create_all, get_engine, get_session
from velorag.db.deps import get_db


def init_db():
    # Ensure all ORM models are registered before creating tables
    import velorag.db.models  # noqa: F401
    import velorag.knowledge.models  # noqa: F401
    import velorag.chat.models  # noqa: F401

    create_all()


__all__ = ["Base", "get_db", "get_engine", "get_session", "init_db"]
