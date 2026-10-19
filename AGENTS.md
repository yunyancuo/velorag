# VeloRAG project context

## User-defined objective

The user stated the founding objective on 2026-10-20:

> 我要用一套新算法解决大知识库检索效果差的问题

VeloRAG is a retrieval algorithm research project targeting retrieval quality in
large knowledge bases. The application, Canvas, citations, and campus examples
support implementation and evaluation. They do not define the research objective.

The user confirmed that no real corpus is available and requires meaningful
algorithmic differentiation from market RAG products. Public benchmarks and
controlled stress tests may be used to develop the idea. Corpus scale, target
domain, languages, and available hardware remain unspecified. Do not treat a
particular scale or failure mechanism as an established requirement or result.

The current proposed research focus and prior-art challenges are documented in
`docs/research/001-differentiation.md`. Its pilot direction remains a hypothesis;
no new retrieval algorithm or large-corpus quality improvement has been proven.

## Repository map

- `src/velorag/retrieval/`: BM25, fusion, graph retrieval, and reranking.
- `src/velorag/indexing/`: vector stores and sharding.
- `src/velorag/pipeline.py`: ingestion, retrieval, and generation orchestration.
- `src/velorag/eval/`: existing small development benchmark and metrics.
- `tests/`: implementation regression tests.
- `frontend/`: React application supporting demonstrations.

## Verified baseline

Commit `1a2441f` fixes graph metadata filtering and frontend production builds.
It passed 33 Python tests, `npm ci --offline`, and `npm run build`.
It was pushed to `codex/takeover-baseline`; draft PR #1 targets `master`.
The built-in benchmark contains five documents and eight queries. It is a
development check, not evidence of large-corpus retrieval performance.

## Research working conventions

- Separate implementation bugs, representation errors, candidate recall losses,
  ranking errors, and evidence coverage failures before attributing results to scale.
- Measure candidate recall before and after reranking. The research branch now
  separates `candidate_k` from final `top_k`, then truncates after reranking.
- Compare improvements with a corrected and appropriately tuned baseline.
- Fix the query/evidence set while varying corpus size and distractor type;
  disclose incomplete relevance labels and accidental relevant distractors.
- Report retrieval quality together with latency, resource use, and index cost.
- Keep development and held-out evaluation separate; preserve raw results.
- Label candidate mechanisms and novelty claims as unverified until tested and
  supported by a documented literature comparison.
- Do not infer scalability or algorithm novelty from HASH-embedding toy examples.
