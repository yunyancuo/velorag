from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

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
        self._dimension = settings.dimension

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.settings.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        if self._model is None:
            _ = self.model
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
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

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx
        api_key = self.settings.api_key
        api_base = self.settings.api_base or "https://api.openai.com/v1"

        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{api_base}/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"input": texts, "model": self.settings.model_name},
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def create_embedder(settings: EmbeddingSettings) -> BaseEmbedder:
    if settings.backend == EmbeddingBackend.SENTENCE_TRANSFORMER:
        return SentenceTransformerEmbedder(settings)
    elif settings.backend == EmbeddingBackend.OPENAI:
        return OpenAIEmbedder(settings)
    raise ValueError(f"Unsupported embedding backend: {settings.backend}")
