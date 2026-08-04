import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api, KBResponse, DocumentResponse } from "../api/client";

const KBDetail: React.FC = () => {
  const { kbId } = useParams<{ kbId: string }>();
  const [kb, setKB] = useState<KBResponse | null>(null);
  const [docs, setDocs] = useState<DocumentResponse[]>([]);
  const [uploading, setUploading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    if (!kbId) return;
    api.getKB(kbId).then(({ data }) => setKB(data)).catch(() => navigate("/"));
    api.listDocuments(kbId).then(({ data }) => setDocs(data));
  }, [kbId]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !kbId) return;
    setUploading(true);
    try {
      await api.uploadDocument(kbId, file);
      const { data } = await api.listDocuments(kbId);
      setDocs(data);
    } catch {}
    setUploading(false);
  };

  if (!kb) return <div>Loading...</div>;

  return (
    <div className="page">
      <div className="page-header">
        <h2>{kb.name}</h2>
        <span className="badge">{kb.retrieval_mode}</span>
      </div>
      <p style={{color:"#666"}}>{kb.description}</p>

      <div className="stats">
        <div className="stat">
          <span className="stat-value">{docs.length}</span>
          <span className="stat-label">Documents</span>
        </div>
        <div className="stat">
          <span className="stat-value">{kb.chunk_strategy}</span>
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
          </div>
        ))}
        {docs.length === 0 && (
          <p style={{color:"#999",textAlign:"center"}}>No documents yet</p>
        )}
      </div>
    </div>
  );
};

export default KBDetail;
