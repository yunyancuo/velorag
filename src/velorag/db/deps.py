from __future__ import annotations

from velorag.db.base import get_session


def get_db():
    session = get_session()
    try:
        yield session
    finally:
        session.close()
