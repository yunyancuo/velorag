import React, { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, DocumentResponse, extractApiError, KBResponse } from "../api/client";

const KBDetail: React.FC = () => {
  const { kbId } = useParams<{ kbId: string }>();
  const [kb, setKB] = useState<KBResponse | null>(null);
  const [docs, setDocs] = useState<DocumentResponse[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    if (!kbId) return;
    setError("");
    api
      .getKB(kbId)
      .then(({ data }) => setKB(data))
      .catch(() => navigate("/"));
    api
      .listDocuments(kbId)
      .then(({ data }) => setDocs(data))
      .catch((err) => setError(extractApiError(err, "Failed to load documents")));
  }, [kbId, navigate]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !kbId) return;
    setUploading(true);
    setError("");
    try {
      await api.uploadDocument(kbId, file);
      const { data } = await api.listDocuments(kbId);
      setDocs(data);
    } catch (err) {
      setError(extractApiError(err, "Document upload failed"));
    } finally {
      e.target.value = "";
      setUploading(false);
    }
  };

  if (!kb) return <div>Loading...</div>;

  return (
    <div className="page">
      <div className="page-header">
        <h2>{kb.name}</h2>
        <span className="badge">{kb.retrieval_mode || "hybrid"}</span>
      </div>
      <p style={{ color: "#8b949e" }}>{kb.description}</p>
      {error && <p className="error">{error}</p>}

      <div className="stats">
        <div className="stat">
          <span className="stat-value">{docs.length}</span>
          <span className="stat-label">Documents</span>
        </div>
        <div className="stat">
          <span className="stat-value">
            {kb.chunk_strategy || "recursive"} ({kb.chunk_size}/{kb.chunk_overlap})
          </span>
          <span className="stat-label">Chunking</span>
        </div>
        <div className="stat">
          <span className="stat-value">{kb.embedding_model}</span>
          <span className="stat-label">Embedding</span>
        </div>
      </div>

      <div className="upload-area">
        <label className="upload-btn">
          {uploading ? "Uploading..." : "Upload Document"}
          <input
            type="file"
            accept=".txt,.md,.pdf,.docx,.markdown"
            onChange={handleUpload}
            disabled={uploading}
            style={{ display: "none" }}
          />
        </label>
      </div>

      <div className="doc-list">
        <h3>Documents</h3>
        {docs.map((doc) => (
          <div key={doc.id} className="doc-item">
            <span className="doc-name">{doc.filename}</span>
            <span className={`doc-status status-${doc.status}`}>
              {doc.status}
            </span>
            <span className="doc-chunks">{doc.chunk_count} chunks</span>
            {doc.error_msg && <span className="error">{doc.error_msg}</span>}
          </div>
        ))}
        {docs.length === 0 && (
          <p style={{ color: "#999", textAlign: "center" }}>No documents yet</p>
        )}
      </div>
    </div>
  );
};

export default KBDetail;
