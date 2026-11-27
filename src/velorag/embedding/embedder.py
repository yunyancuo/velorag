from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Optional

from velorag.core.security import validate_http_url
from velorag.core.settings import EmbeddingSettings
from velorag.core.types import EmbeddingBackend


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings
        self._model = None
        self._fallback: Optional[HashEmbedder] = None
        self._dimension = settings.dimension

    @property
    def model(self):
        if self._model is None and self._fallback is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.settings.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
            except Exception:
                self._fallback = HashEmbedder(self.settings)
        return self._model

    @property
    def dimension(self) -> int:
        if self._model is None and self._fallback is None:
            _ = self.model
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        m = self.model
        if self._fallback is not None or m is None:
            return self._fallback.embed(texts)
        embeddings = m.encode(
            texts,
            batch_size=self.settings.batch_size,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings
        self._dimension = settings.dimension
        self._api_base = validate_http_url(
            settings.api_base or "https://api.openai.com/v1"
        ).rstrip("/")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        api_key = self.settings.api_key
        if not api_key:
            raise ValueError("使用 OpenAIEmbedder 必须配置 api_key")

        import httpx

        batch_size = max(1, int(self.settings.batch_size or 64))
        vectors: list[list[float]] = []

        with httpx.Client(timeout=60) as client:
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                response = client.post(
                    f"{self._api_base}/embeddings",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"input": batch, "model": self.settings.model_name},
                )
                response.raise_for_status()
                data = response.json()
                vectors.extend(item["embedding"] for item in data["data"])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HashEmbedder(BaseEmbedder):
    """零外部依赖的确定性字符/词元 n-gram 特征哈希向量化器，适用于离线与测试环境。"""

    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings
        self._dimension = max(1, int(settings.dimension or 384))

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        if not text:
            return vec

        lower = text.lower()
        tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lower)
        features = list(tokens)
        for i in range(len(tokens) - 1):
            features.append(f"{tokens[i]}_{tokens[i + 1]}")
        clean = re.sub(r"\s+", "", lower)
        for i in range(len(clean) - 2):
            features.append(clean[i : i + 3])

        if not features:
            return vec

        for feat in features:
            digest = hashlib.md5(feat.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self._dimension
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-10:
            vec = [v / norm for v in vec]
        return vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [self._encode_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode_one(text)


def create_embedder(settings: EmbeddingSettings) -> BaseEmbedder:
    if settings.backend == EmbeddingBackend.HASH:
        return HashEmbedder(settings)
    if settings.backend == EmbeddingBackend.SENTENCE_TRANSFORMER:
        return SentenceTransformerEmbedder(settings)
    if settings.backend == EmbeddingBackend.OPENAI:
        return OpenAIEmbedder(settings)
    raise ValueError(f"Unsupported embedding backend: {settings.backend}")
