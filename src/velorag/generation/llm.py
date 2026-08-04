from __future__ import annotations

from typing import Optional, AsyncIterator

from velorag.core.types import Chunk, GenerationRequest, GenerationResponse, SearchResult


DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's question based on the provided context.
If the context does not contain sufficient information to answer the question, say so honestly.
Always cite sources when using information from the context."""


def build_context_prompt(query: str, chunks: list[Chunk], max_tokens: int = 4096) -> str:
    context_parts = []
    token_estimate = 0

    for i, chunk in enumerate(chunks):
        part = f"[Source {i + 1}] {chunk.text}"
        token_estimate += len(part) // 4
        if token_estimate > max_tokens:
            break
        context_parts.append(part)

    context = "\n\n".join(context_parts)

    return f"""Context information:
{context}

Question: {query}

Answer:"""


class LLMGenerator:
    def __init__(self, api_base: str | None = None, api_key: str | None = None, model: str = "gpt-3.5-turbo"):
        import os
        self.api_base = api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.system_prompt = DEFAULT_SYSTEM_PROMPT

    def generate(self, request: GenerationRequest, results: list[SearchResult]) -> GenerationResponse:
        chunks = [r.chunk for r in results] if results else request.context_chunks
        prompt = build_context_prompt(request.query, chunks, request.max_tokens // 2)

        messages = [
            {"role": "system", "content": request.system_prompt or self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        import httpx
        with httpx.Client(timeout=120) as client:
            response = client.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        token_count = data.get("usage", {}).get("total_tokens", 0)

        return GenerationResponse(
            content=content,
            sources=results,
            token_count=token_count,
        )

    async def generate_stream(
        self, request: GenerationRequest, results: list[SearchResult]
    ) -> AsyncIterator[str]:
        chunks = [r.chunk for r in results] if results else request.context_chunks
        prompt = build_context_prompt(request.query, chunks, request.max_tokens // 2)

        messages = [
            {"role": "system", "content": request.system_prompt or self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        import httpx
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        import json
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
