from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from velorag.core.types import (
    ChunkStrategy,
    EmbeddingBackend,
    VectorStoreBackend,
    RetrievalMode,
    RerankerType,
)


@dataclass
class ChunkSettings:
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 64
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ".", " ", ""])


@dataclass
class EmbeddingSettings:
    backend: EmbeddingBackend = EmbeddingBackend.SENTENCE_TRANSFORMER
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 64
    api_key: Optional[str] = None
    api_base: Optional[str] = None


@dataclass
class IndexSettings:
    backend: VectorStoreBackend = VectorStoreBackend.CHROMA
    persist_dir: str = "./data/velorag"
    collection_name: str = "default"
    num_shards: int = 1
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100


@dataclass
class RetrievalSettings:
    default_mode: RetrievalMode = RetrievalMode.HYBRID
    dense_top_k: int = 50
    sparse_top_k: int = 50
    hybrid_top_k: int = 10
    fusion_alpha: float = 0.5
    reranker: Optional[RerankerType] = None
    min_score_threshold: float = 0.0


@dataclass
class GraphSettings:
    enabled: bool = True
    max_entity_distance: int = 2
    entity_weight: float = 0.3
    relation_types: list[str] = field(default_factory=lambda: [
        "related_to", "part_of", "causes", "depends_on"
    ])


@dataclass
class CacheSettings:
    backend: str = "memory"
    redis_url: Optional[str] = None
    ttl_seconds: int = 300
    max_entries: int = 10000


@dataclass
class ServerSettings:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class DBSettings:
    url: str = "sqlite:///./data/velorag.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class AuthSettings:
    secret_key: str = "velorag-dev-secret-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    api_key_prefix: str = "vr-"


@dataclass
class CelerySettings:
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/1"
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: list[str] = field(default_factory=lambda: ["json"])
    timezone: str = "UTC"
    enable_utc: bool = True


@dataclass
class Settings:
    chunk: ChunkSettings = field(default_factory=ChunkSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    index: IndexSettings = field(default_factory=IndexSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    graph: GraphSettings = field(default_factory=GraphSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    db: DBSettings = field(default_factory=DBSettings)
    auth: AuthSettings = field(default_factory=AuthSettings)
    celery: CelerySettings = field(default_factory=CelerySettings)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            chunk=ChunkSettings(
                strategy=ChunkStrategy(os.getenv("VELORAG_CHUNK_STRATEGY", "recursive")),
                chunk_size=int(os.getenv("VELORAG_CHUNK_SIZE", "512")),
                chunk_overlap=int(os.getenv("VELORAG_CHUNK_OVERLAP", "64")),
            ),
            embedding=EmbeddingSettings(
                backend=EmbeddingBackend(os.getenv("VELORAG_EMBED_BACKEND", "sentence_transformer")),
                model_name=os.getenv("VELORAG_EMBED_MODEL", "all-MiniLM-L6-v2"),
                api_key=os.getenv("OPENAI_API_KEY"),
                api_base=os.getenv("OPENAI_API_BASE"),
            ),
            index=IndexSettings(
                backend=VectorStoreBackend(os.getenv("VELORAG_INDEX_BACKEND", "chroma")),
                persist_dir=os.getenv("VELORAG_DATA_DIR", "./data/velorag"),
                num_shards=int(os.getenv("VELORAG_NUM_SHARDS", "1")),
            ),
            db=DBSettings(
                url=os.getenv("VELORAG_DB_URL", "sqlite:///./data/velorag.db"),
                echo=os.getenv("VELORAG_DB_ECHO", "").lower() == "true",
            ),
            auth=AuthSettings(
                secret_key=os.getenv("VELORAG_SECRET_KEY", "velorag-dev-secret-change-in-production"),
                access_token_expire_minutes=int(os.getenv("VELORAG_TOKEN_EXPIRE", "60")),
            ),
            celery=CelerySettings(
                broker_url=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
                result_backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
            ),
        )


# Module-level singleton (frozen after first load)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def configure_settings(settings: Settings):
    global _settings
    _settings = settings
