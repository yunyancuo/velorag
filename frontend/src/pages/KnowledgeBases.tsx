import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, extractApiError, KBCreate, KBResponse } from "../api/client";

const KnowledgeBases: React.FC = () => {
  const [kbs, setKBs] = useState<KBResponse[]>([]);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [error, setError] = useState("");

  const load = async () => {
    try {
      setError("");
      const { data } = await api.listKB();
      setKBs(data);
    } catch (err) {
      setError(extractApiError(err, "Failed to load knowledge bases"));
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    try {
      setError("");
      const payload: KBCreate = { name: name.trim(), description: desc.trim() };
      await api.createKB(payload);
      setName("");
      setDesc("");
      setCreating(false);
      await load();
    } catch (err) {
      setError(extractApiError(err, "Failed to create knowledge base"));
    }
  };

  const handleDelete = async (id: string) => {
    try {
      setError("");
      await api.deleteKB(id);
      await load();
    } catch (err) {
      setError(extractApiError(err, "Failed to delete knowledge base"));
    }
  };

  return (
    <div className="page">
      <div className="page-header">
        <h2>Knowledge Bases</h2>
        <button onClick={() => setCreating(true)}>+ New</button>
      </div>

      {error && <p className="error">{error}</p>}

      {creating && (
        <form className="kb-form" onSubmit={handleCreate}>
          <input
            type="text"
            placeholder="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
          />
          <input
            type="text"
            placeholder="Description"
            value={desc}
            onChange={(e) => setDesc(e.target.value)}
          />
          <div style={{ display: "flex", gap: 8 }}>
            <button type="submit">Create</button>
            <button type="button" onClick={() => setCreating(false)}>
              Cancel
            </button>
          </div>
        </form>
      )}

      <div className="kb-grid">
        {kbs.map((kb) => (
          <div key={kb.id} className="kb-card">
            <Link to={`/kb/${kb.id}`}>
              <h3>{kb.name}</h3>
            </Link>
            <p>{kb.description}</p>
            <div className="kb-meta">
              <span>{kb.document_count ?? 0} documents</span>
              <span>{kb.retrieval_mode || "hybrid"}</span>
              <span>{kb.is_public ? "public" : "private"}</span>
            </div>
            <div className="kb-actions">
              <Link to={`/search/${kb.id}`}>
                <button className="btn-sm">Search</button>
              </Link>
              {kb.id !== "default" && (
                <button
                  className="btn-sm btn-danger"
                  onClick={() => handleDelete(kb.id)}
                >
                  Delete
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default KnowledgeBases;
