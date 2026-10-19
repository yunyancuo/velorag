import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api, GraphSubgraph, QueryResponse, SearchResult } from "../api/client";

const SAMPLE_QUERIES = [
  "硕士研究生毕业总学分和学位必修课分别要求不少于多少学分？",
  "AI开源实验室算力集群使用什么调度系统？train训练队列最长运行多少小时？",
  "医疗AI联合培养项目中，医学影像数据在训练前必须完成什么处理？",
  "How does VeloRAG fuse dense vector search, BM25, and knowledge graph retrieval?",
  "请问紫金港校区第三食堂的夜宵麻辣烫几点关门？（无依据拒答测试）",
];

const Search: React.FC = () => {
  const { kbId } = useParams<{ kbId: string }>();
  const [query, setQuery] = useState(SAMPLE_QUERIES[0]);
  const [mode, setMode] = useState("graph_enhanced");
  const [qaResult, setQaResult] = useState<QueryResponse | null>(null);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [graph, setGraph] = useState<GraphSubgraph | null>(null);
  const [loading, setLoading] = useState(false);

  const loadGraph = async (q?: string) => {
    try {
      const { data } = await api.getGraph(q, 24);
      setGraph(data);
    } catch {}
  };

  useEffect(() => {
    loadGraph();
  }, []);

  const handleRun = async (e?: React.FormEvent, customQuery?: string) => {
    if (e) e.preventDefault();
    const targetQuery = (customQuery ?? query).trim();
    if (!targetQuery) return;
    setLoading(true);
    try {
      const searchMode = mode === "graph_enhanced" ? "hybrid" : mode;
      const useGraph = mode === "graph_enhanced" || mode === "hybrid";
      const [qaRes, searchRes] = await Promise.all([
        api.query(targetQuery, 5, searchMode, useGraph),
        api.search(kbId || "default", targetQuery, 5, searchMode, useGraph),
      ]);
      setQaResult(qaRes.data);
      setSearchResults(searchRes.data.results);
      await loadGraph(targetQuery);
    } catch {}
    setLoading(false);
  };

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h2>Grounded QA & Hybrid Knowledge Graph Studio</h2>
          <p style={{ color: "#94a3b8", fontSize: "0.9rem", marginTop: 4 }}>
            Multilingual Dense + BM25 + Multi-Hop Knowledge Graph Retrieval with Inline [Source N] Citations & Refusal Guardrail
          </p>
        </div>
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 14 }}>
        {SAMPLE_QUERIES.map((sq, i) => (
          <button
            key={i}
            type="button"
            className="chip-btn"
            onClick={() => {
              const cleaned = sq.replace("（无依据拒答测试）", "");
              setQuery(cleaned);
              handleRun(undefined, cleaned);
            }}
          >
            {sq}
          </button>
        ))}
      </div>

      <form className="search-form" onSubmit={(e) => handleRun(e)}>
        <input
          type="text"
          placeholder="Ask any question in Chinese or English..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value)}
          className="mode-select"
        >
          <option value="graph_enhanced">Graph-Enhanced Hybrid (RRF + KG)</option>
          <option value="hybrid">Hybrid (Dense + BM25)</option>
          <option value="dense">Dense Vector Only</option>
          <option value="sparse">Sparse BM25 Only</option>
        </select>
        <button type="submit" disabled={loading}>
          {loading ? "Running..." : "Ask & Verify"}
        </button>
      </form>

      {qaResult && (
        <div className="qa-card">
          <div className="qa-header">
            <span style={{ fontWeight: 600 }}>Grounded Answer</span>
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <span
                className={qaResult.refused ? "badge-danger" : "badge-success"}
              >
                {qaResult.refused ? "🛡️ Refused (Insufficient Evidence)" : "✅ Evidence Verified"}
              </span>
              <span className="badge-neutral">
                Confidence: {(qaResult.confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>
          <p className="qa-answer">{qaResult.answer}</p>

          {qaResult.citations.length > 0 && (
            <div style={{ marginTop: 14 }}>
              <h4 style={{ fontSize: "0.85rem", color: "#94a3b8", marginBottom: 8 }}>
                Verified Inline Citations ({qaResult.citations.length})
              </h4>
              <div className="citation-grid">
                {qaResult.citations.map((c) => (
                  <div key={c.index} className="citation-item">
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                      <strong style={{ color: "#38bdf8" }}>
                        [Source {c.index}] {c.source}
                      </strong>
                      <span style={{ fontSize: "0.78rem", color: "#94a3b8" }}>
                        score: {c.score.toFixed(3)}
                      </span>
                    </div>
                    <div style={{ fontSize: "0.84rem", color: "#cbd5e1" }}>{c.quote}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      <div className="two-col-grid">
        <div>
          <h3 style={{ marginBottom: 12, fontSize: "1.05rem" }}>
            Retrieved Context Chunks ({searchResults.length})
          </h3>
          <div className="results">
            {searchResults.map((r, idx) => (
              <div key={r.chunk_id || idx} className="result-card">
                <div className="result-header">
                  <span className="result-source">
                    #{idx + 1} · {r.source} ({r.retrieval_source || mode})
                  </span>
                  <span className="result-score">score: {r.score.toFixed(4)}</span>
                </div>
                <p className="result-text">{r.text}</p>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 style={{ marginBottom: 12, fontSize: "1.05rem" }}>
            Knowledge Graph Subgraph ({graph?.nodes.length || 0} entities, {graph?.edges.length || 0} relations)
          </h3>
          <div className="result-card">
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 12 }}>
              {graph?.nodes.slice(0, 18).map((n) => (
                <span key={n.id} className="entity-pill">
                  {n.label} <small>({n.entity_type})</small>
                </span>
              ))}
            </div>
            <div style={{ fontSize: "0.83rem", color: "#94a3b8" }}>
              {graph?.edges.slice(0, 10).map((e, idx) => (
                <div key={idx} style={{ padding: "4px 0", borderBottom: "1px solid #1e293b" }}>
                  <code style={{ color: "#e2e8f0" }}>{e.source}</code>
                  <span style={{ color: "#38bdf8", margin: "0 8px" }}>—[{e.relation}]→</span>
                  <code style={{ color: "#e2e8f0" }}>{e.target}</code>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Search;
