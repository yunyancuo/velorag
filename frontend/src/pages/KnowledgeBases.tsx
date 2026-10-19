import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, KBResponse, KBCreate } from "../api/client";

const KnowledgeBases: React.FC = () => {
  const [kbs, setKBs] = useState<KBResponse[]>([]);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");

  const load = async () => {
    try {
      const { data } = await api.listKB();
      setKBs(data);
    } catch {}
  };

  useEffect(() => { load(); }, []);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    const payload: KBCreate = { name: name.trim(), description: desc.trim() };
    await api.createKB(payload);
    setName("");
    setDesc("");
    setCreating(false);
    load();
  };

  const handleDelete = async (id: string) => {
    await api.deleteKB(id);
    load();
  };

  return (
    <div className="page">
      <div className="page-header">
        <h2>Knowledge Bases</h2>
        <button onClick={() => setCreating(true)}>+ New</button>
      </div>

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
              <span>{kb.chunk_size} characters per chunk</span>
              <span>{kb.embedding_model}</span>
            </div>
            <div className="kb-actions">
              <Link to={`/search/${kb.id}`}>
                <button className="btn-sm">Search</button>
              </Link>
              <button
                className="btn-sm btn-danger"
                onClick={() => handleDelete(kb.id)}
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default KnowledgeBases;
