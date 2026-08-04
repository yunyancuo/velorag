from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AlembicConfig:
    script_location: str = "alembic"
    sqlalchemy_url: str = "sqlite:///./data/velorag.db"
    target_metadata: object = None
