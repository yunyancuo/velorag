from __future__ import annotations

import os
import re
from typing import Optional, AsyncIterator

from velorag.core.types import (
    Chunk,
    Citation,
    GenerationRequest,
    GenerationResponse,
    SearchResult,
)
from velorag.retrieval.engine import tokenize_multilingual


DEFAULT_SYSTEM_PROMPT = """You are a grounded, trustworthy AI research assistant powered by VeloRAG.
Answer the user's question ONLY using the provided context sources.
Rules:
1. Cite supporting sources inline using [Source N] format (e.g., [Source 1], [Source 2]).
2. If the provided context does not contain enough evidence to answer the question, or if sources conflict/are outdated, explicitly refuse to guess and state what information is missing.
3. Never fabricate facts, numbers, or citations."""

REFUSAL_PHRASES = (
    "insufficient information",
    "not contain sufficient",
    "cannot answer",
    "do not have enough information",
    "no relevant information",
    "资料不足",
    "无法回答",
    "未提供相关信息",
    "没有找到相关依据",
    "无法从提供的资料中",
)


def build_context_prompt(query: str, chunks: list[Chunk], max_tokens: int = 4096) -> str:
    context_parts = []
    token_estimate = 0

    for i, chunk in enumerate(chunks):
        source_name = chunk.metadata.get("source") or chunk.metadata.get("filename") or chunk.doc_id or f"doc_{i + 1}"
        part = f"[Source {i + 1}] (source={source_name}, chunk_id={chunk.id})\n{chunk.text}"
        token_estimate += max(1, len(part) // 4)
        if token_estimate > max_tokens:
            break
        context_parts.append(part)

    context = "\n\n".join(context_parts)

    return f"""Context information:
{context}

Question: {query}

Provide a concise, well-structured answer with [Source N] citations, or clearly state if the context is insufficient:"""


def extract_citations_from_answer(
    answer: str, chunks: list[Chunk], results: list[SearchResult]
) -> list[Citation]:
    cited_indices: list[int] = []
    for match in re.finditer(r"\[(?:Source|Citation|来源)\s*(\d+)\]", answer, re.IGNORECASE):
        idx = int(match.group(1))
        if 1 <= idx <= len(chunks) and idx not in cited_indices:
            cited_indices.append(idx)

    score_map = {r.chunk.id: r.score for r in results}
    citations: list[Citation] = []
    for idx in cited_indices:
        chunk = chunks[idx - 1]
        source_name = (
            chunk.metadata.get("source")
            or chunk.metadata.get("filename")
            or chunk.doc_id
            or f"source_{idx}"
        )
        quote = chunk.text.strip()
        if len(quote) > 220:
            quote = quote[:217] + "..."
        citations.append(
            Citation(
                index=idx,
                chunk_id=chunk.id,
                doc_id=chunk.doc_id,
                source=str(source_name),
                quote=quote,
                score=float(score_map.get(chunk.id, 0.0)),
                metadata=dict(chunk.metadata),
            )
        )
    return citations


def check_evidence_sufficiency(
    query: str, chunks: list[Chunk], results: list[SearchResult], min_score: float = 0.12
) -> tuple[bool, Optional[str], float]:
    if not chunks:
        return False, "No supporting chunks retrieved from the knowledge base.", 0.0

    best_score = max((r.score for r in results), default=0.5)
    q_tokens = set(tokenize_multilingual(query))
    stopwords = {
        "what", "how", "why", "when", "where", "who", "which", "is", "are", "the", "a", "an",
        "in", "of", "to", "for", "and", "or", "can", "does", "do",
        "什么", "如何", "怎么", "为什么", "哪里", "哪个", "请问", "的", "了", "吗", "呢", "是", "在", "有",
    }
    meaningful_q = {t for t in q_tokens if t not in stopwords and len(t) >= 2}
    if not meaningful_q:
        meaningful_q = q_tokens

    doc_tokens: set[str] = set()
    for c in chunks:
        doc_tokens.update(tokenize_multilingual(c.text))

    overlap_ratio = len(meaningful_q & doc_tokens) / max(1, len(meaningful_q))
    top_chunk_tokens = set(tokenize_multilingual(chunks[0].text))
    top_overlap_ratio = len(meaningful_q & top_chunk_tokens) / max(1, len(meaningful_q))
    confidence = round(0.55 * min(1.0, best_score) + 0.45 * min(1.0, top_overlap_ratio * 1.5), 4)

    if results and best_score < min_score:
        return (
            False,
            f"Top retrieval score ({best_score:.3f}) is below minimum evidence threshold ({min_score:.3f}).",
            confidence,
        )
    if meaningful_q and top_overlap_ratio < 0.15:
        return (
            False,
            f"Top evidence chunk has insufficient key-term coverage ({top_overlap_ratio:.2%}).",
            0.08,
        )
    return True, None, max(0.2, confidence)


class LLMGenerator:
    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_base = api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("VELORAG_LLM_MODEL", "gpt-4o-mini")
        self.system_prompt = DEFAULT_SYSTEM_PROMPT

    def _extractive_fallback(
        self, query: str, chunks: list[Chunk], results: list[SearchResult]
    ) -> str:
        """Deterministic extractive synthesis with [Source N] citations when no external LLM API key is configured."""
        q_tokens = set(tokenize_multilingual(query))
        scored_sentences: list[tuple[float, int, str]] = []

        for idx, chunk in enumerate(chunks[:5], start=1):
            parts = re.split(r"(?<=[。！？.!?\n])\s*", chunk.text.strip())
            for sent in parts:
                s = sent.strip()
                if not s:
                    continue
                s_tokens = set(tokenize_multilingual(s))
                overlap = len(q_tokens & s_tokens) / max(1, len(q_tokens))
                if overlap > 0:
                    rank_bonus = 0.5 if idx == 1 else (0.1 / idx)
                    scored_sentences.append((overlap + rank_bonus, idx, s))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top1_sents = [item for item in scored_sentences if item[1] == 1]
        if top1_sents:
            selected = top1_sents[:3]
            covered_tokens: set[str] = set()
            for _, _, s in selected:
                covered_tokens.update(tokenize_multilingual(s))
            top1_doc_id = chunks[0].doc_id

            for score, idx, sent in scored_sentences:
                if idx == 1:
                    continue
                sent_tokens = set(tokenize_multilingual(sent))
                uncovered = (q_tokens - covered_tokens) & sent_tokens
                if not uncovered:
                    continue
                same_doc = chunks[idx - 1].doc_id == top1_doc_id
                has_tech_term = any(
                    len(t) >= 3 or (t.isascii() and t.isalnum() and len(t) >= 2)
                    for t in uncovered
                )
                if same_doc or has_tech_term:
                    selected.append((score, idx, sent))
                    covered_tokens.update(sent_tokens)
        elif scored_sentences:
            selected = scored_sentences[:2]
        else:
            selected = [(1.0, 1, chunks[0].text.strip())]

        lines = []
        seen = set()
        for _, src_idx, sent in selected:
            if sent in seen:
                continue
            seen.add(sent)
            lines.append(f"{sent} [Source {src_idx}]")

        return " ".join(lines)

    def generate(
        self, request: GenerationRequest, results: list[SearchResult]
    ) -> GenerationResponse:
        chunks = [r.chunk for r in results] if results else request.context_chunks

        sufficient, reason, confidence = check_evidence_sufficiency(
            request.query, chunks, results, min_score=request.min_relevance_score
        )
        if not sufficient:
            is_zh = bool(re.search(r"[一-鿿]", request.query))
            refusal_msg = (
                f"抱歉，当前知识库资料不足以回答该问题（原因：{reason}）。请补充相关授权资料后再试。"
                if is_zh
                else f"I cannot answer this question based on the available knowledge base ({reason})."
            )
            return GenerationResponse(
                content=refusal_msg,
                sources=results,
                citations=[],
                token_count=len(refusal_msg) // 4,
                refused=True,
                refusal_reason=reason,
                confidence=confidence,
            )

        content: str = ""
        token_count: int = 0

        if self.api_key:
            try:
                import httpx

                prompt = build_context_prompt(request.query, chunks, request.max_tokens // 2)
                messages = [
                    {"role": "system", "content": request.system_prompt or self.system_prompt},
                    {"role": "user", "content": prompt},
                ]
                with httpx.Client(timeout=60) as client:
                    response = client.post(
                        f"{self.api_base.rstrip('/')}/chat/completions",
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
            except Exception:
                content = self._extractive_fallback(request.query, chunks, results)
                token_count = len(content) // 4
        else:
            content = self._extractive_fallback(request.query, chunks, results)
            token_count = max(1, len(content) // 4)

        lower_content = content.lower()
        llm_refused = any(p in lower_content for p in REFUSAL_PHRASES)
        citations = extract_citations_from_answer(content, chunks, results)
        if not citations and not llm_refused and chunks and request.strict_citation:
            content = f"{content.rstrip()} [Source 1]"
            citations = extract_citations_from_answer(content, chunks, results)

        return GenerationResponse(
            content=content,
            sources=results,
            citations=citations,
            token_count=token_count,
            refused=llm_refused,
            refusal_reason="LLM indicated insufficient context." if llm_refused else None,
            confidence=0.15 if llm_refused else confidence,
        )

    async def generate_stream(
        self, request: GenerationRequest, results: list[SearchResult]
    ) -> AsyncIterator[str]:
        response = self.generate(request, results)
        words = re.findall(r"\S+\s*", response.content)
        for w in words:
            yield w
