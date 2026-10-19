import React, { useEffect, useState } from "react";
import { AgentCanvasModel, api, CanvasRunResponse } from "../api/client";

const NODE_COLORS: Record<string, string> = {
  begin: "#10b981",
  retrieval: "#3b82f6",
  rerank: "#8b5cf6",
  switch: "#f59e0b",
  code: "#ec4899",
  tool: "#14b8a6",
  generate: "#06b6d4",
  end: "#64748b",
};

const AgentCanvasPage: React.FC = () => {
  const [canvas, setCanvas] = useState<AgentCanvasModel | null>(null);
  const [query, setQuery] = useState("硕士研究生毕业总学分要求不少于多少学分？");
  const [runResult, setRunResult] = useState<CanvasRunResponse | null>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    api.listCanvases().then(({ data }) => {
      if (data.length > 0) setCanvas(data[0]);
    });
  }, []);

  const handleRun = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canvas || !query.trim()) return;
    setRunning(true);
    try {
      const { data } = await api.runCanvas(canvas.id, query);
      setRunResult(data);
    } catch {}
    setRunning(false);
  };

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h2>Agent Canvas DAG Workflow Studio</h2>
          <p style={{ color: "#94a3b8", fontSize: "0.9rem", marginTop: 4 }}>
            Visual DAG pipeline orchestration for Retrieval, Reranking, Evidence Guardrail Switching & Grounded Generation
          </p>
        </div>
      </div>

      {canvas && (
        <div className="canvas-board">
          <div style={{ marginBottom: 12, display: "flex", justifyContent: "space-between" }}>
            <strong>{canvas.name}</strong>
            <span style={{ fontSize: "0.82rem", color: "#94a3b8" }}>
              ID: {canvas.id} · {canvas.nodes.length} Nodes · {canvas.edges.length} Edges
            </span>
          </div>
          <div className="dag-flow">
            {canvas.nodes.map((node, index) => (
              <React.Fragment key={node.id}>
                <div
                  className="dag-node"
                  style={{ borderTop: `3px solid ${NODE_COLORS[node.type] || "#38bdf8"}` }}
                >
                  <div className="dag-node-type" style={{ color: NODE_COLORS[node.type] || "#38bdf8" }}>
                    {node.type.toUpperCase()}
                  </div>
                  <div className="dag-node-title">{node.name}</div>
                  <div className="dag-node-id">id: {node.id}</div>
                  {Object.keys(node.config).length > 0 && (
                    <div className="dag-node-config">
                      {Object.entries(node.config).map(([k, v]) => (
                        <div key={k}>
                          <span>{k}:</span> <code>{String(v)}</code>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                {index < canvas.nodes.length - 1 && <div className="dag-arrow">→</div>}
              </React.Fragment>
            ))}
          </div>
        </div>
      )}

      <form className="search-form" onSubmit={handleRun} style={{ marginTop: 20 }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter test query to execute through the Agent Canvas DAG..."
        />
        <button type="submit" disabled={running}>
          {running ? "Executing DAG..." : "▶ Execute Canvas Workflow"}
        </button>
      </form>

      {runResult && (
        <div className="two-col-grid" style={{ marginTop: 18 }}>
          <div className="result-card">
            <h3 style={{ marginBottom: 10, fontSize: "1rem" }}>
              Execution Trace ({runResult.trace.length} steps)
            </h3>
            {runResult.trace.map((t) => (
              <div
                key={t.step}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "8px 0",
                  borderBottom: "1px solid #1e293b",
                  fontSize: "0.88rem",
                }}
              >
                <span>
                  <strong>Step {t.step}:</strong> {t.node_name}{" "}
                  <small style={{ color: "#94a3b8" }}>({t.node_type})</small>
                </span>
                <span style={{ color: "#38bdf8" }}>{t.elapsed_ms} ms</span>
              </div>
            ))}
            <div style={{ marginTop: 12, fontSize: "0.83rem", color: "#94a3b8" }}>
              Context Variables: <code>{JSON.stringify(runResult.variables)}</code>
            </div>
          </div>

          <div className="result-card">
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
              <h3 style={{ fontSize: "1rem" }}>Workflow Output</h3>
              <span className={runResult.refused ? "badge-danger" : "badge-success"}>
                {runResult.refused ? "Refused" : "Grounded"}
              </span>
            </div>
            <p style={{ lineHeight: 1.6, fontSize: "0.93rem" }}>{runResult.final_answer}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default AgentCanvasPage;
