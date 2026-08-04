import React, { useState } from "react";
import { useParams } from "react-router-dom";
import { api, SearchResult } from "../api/client";

const Search: React.FC = () => {
  const { kbId } = useParams<{ kbId: string }>();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || !kbId) return;
    setLoading(true);
    try {
      const { data } = await api.search(kbId, query);
      setResults(data.results);
      setElapsed(data.elapsed_ms);
    } catch {}
    setLoading(false);
  };

  return (
    <div className="page">
      <div className="page-header">
        <h2>Search</h2>
      </div>

      <form className="search-form" onSubmit={handleSearch}>
        <input
          type="text"
          placeholder="Ask anything..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit" disabled={loading}>
          {loading ? "Searching..." : "Search"}
        </button>
      </form>

      {elapsed > 0 && (
        <p className="search-info">
          {results.length} results in {elapsed}ms
        </p>
      )}

      <div className="results">
        {results.map((r) => (
          <div key={r.chunk_id} className="result-card">
            <div className="result-header">
              <span className="result-source">{r.source}</span>
              <span className="result-score">{(r.score * 100).toFixed(1)}%</span>
            </div>
            <p className="result-text">{r.text}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Search;
