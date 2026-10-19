import React, { useState } from "react";
import { api, EvalModeReport } from "../api/client";

const EvalDashboard: React.FC = () => {
  const [reports, setReports] = useState<EvalModeReport[]>([]);
  const [markdown, setMarkdown] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedMode, setSelectedMode] = useState<string>("graph_enhanced");

  const runSuite = async () => {
    setLoading(true);
    try {
      const { data } = await api.runEvaluation(["dense", "sparse", "hybrid", "graph_enhanced"], 3);
      setReports(data.modes);
      setMarkdown(data.markdown_table);
    } catch {}
    setLoading(false);
  };

  const activeReport = reports.find((r) => r.mode === selectedMode) || reports[0];

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h2>RAG Retrieval, Citation & Refusal Benchmark Suite</h2>
          <p style={{ color: "#94a3b8", fontSize: "0.9rem", marginTop: 4 }}>
            Automated evaluation on Campus & AI Lab Bilingual Corpus (Recall@3, MRR@3, NDCG@3, Citation Precision & Refusal Accuracy)
          </p>
        </div>
        <button className="btn-primary" onClick={runSuite} disabled={loading}>
          {loading ? "Running Benchmark..." : "🚀 Run Full Evaluation Suite"}
        </button>
      </div>

      {reports.length === 0 && !loading && (
        <div className="result-card" style={{ textAlign: "center", padding: "36px 20px" }}>
          <p style={{ color: "#94a3b8", marginBottom: 14 }}>
            Click "Run Full Evaluation Suite" to benchmark Dense, Sparse BM25, Hybrid RRF, and Graph-Enhanced Retrieval across factual and unanswerable trap queries.
          </p>
          <button className="btn-primary" onClick={runSuite}>
            Start Benchmark Now
          </button>
        </div>
      )}

      {reports.length > 0 && (
        <>
          <div className="result-card" style={{ marginBottom: 18 }}>
            <h3 style={{ marginBottom: 12, fontSize: "1rem" }}>Comparative Mode Summary</h3>
            <table className="eval-table">
              <thead>
                <tr>
                  <th>Mode</th>
                  <th>Recall@3</th>
                  <th>MRR@3</th>
                  <th>NDCG@3</th>
                  <th>Citation Precision</th>
                  <th>Keyword Hit</th>
                  <th>Refusal Accuracy</th>
                  <th>Avg Latency</th>
                </tr>
              </thead>
              <tbody>
                {reports.map((rep) => {
                  const m = rep.metrics;
                  return (
                    <tr
                      key={rep.mode}
                      onClick={() => setSelectedMode(rep.mode)}
                      style={{ cursor: "pointer", background: rep.mode === selectedMode ? "#1e293b" : "transparent" }}
                    >
                      <td><code>{rep.mode}</code></td>
                      <td>{(m.recall_at_k * 100).toFixed(1)}%</td>
                      <td>{(m.mrr_at_k * 100).toFixed(1)}%</td>
                      <td>{(m.ndcg_at_k * 100).toFixed(1)}%</td>
                      <td>{(m.citation_precision * 100).toFixed(1)}%</td>
                      <td>{(m.keyword_coverage * 100).toFixed(1)}%</td>
                      <td>{(m.refusal_accuracy * 100).toFixed(1)}%</td>
                      <td>{m.avg_latency_ms.toFixed(2)} ms</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {activeReport && (
            <div className="result-card">
              <h3 style={{ marginBottom: 12, fontSize: "1rem" }}>
                Per-Query Breakdown ({activeReport.mode})
              </h3>
              <table className="eval-table">
                <thead>
                  <tr>
                    <th>Query</th>
                    <th>Category</th>
                    <th>Recall@3</th>
                    <th>Citations</th>
                    <th>Refusal Check</th>
                    <th>Answer Preview</th>
                  </tr>
                </thead>
                <tbody>
                  {activeReport.details.map((d) => (
                    <tr key={d.id}>
                      <td style={{ maxWidth: 280 }}>{d.query}</td>
                      <td><code>{d.category}</code></td>
                      <td>{(d.recall_at_k * 100).toFixed(0)}%</td>
                      <td>{d.citations_count}</td>
                      <td>
                        <span className={d.refusal_correct ? "badge-success" : "badge-danger"}>
                          {d.refused ? "Refused" : "Answered"} ({d.refusal_correct ? "PASS" : "FAIL"})
                        </span>
                      </td>
                      <td style={{ maxWidth: 320, fontSize: "0.82rem", color: "#cbd5e1" }}>
                        {d.answer_preview}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {markdown && (
            <div className="result-card" style={{ marginTop: 18 }}>
              <h4 style={{ marginBottom: 8, fontSize: "0.9rem", color: "#94a3b8" }}>
                Markdown Report (Ready for GitHub Issue / PR)
              </h4>
              <pre style={{ fontSize: "0.82rem", overflowX: "auto", color: "#e2e8f0" }}>
                {markdown}
              </pre>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default EvalDashboard;
