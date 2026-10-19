import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "/api/v1";

const client = axios.create({
  baseURL: API_BASE,
  headers: { "Content-Type": "application/json" },
});

client.interceptors.request.use((config) => {
  const token = localStorage.getItem("velorag_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export interface LoginRequest {
  username: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user_id: string;
  username: string;
}

export interface KBResponse {
  id: string;
  name: string;
  description: string;
  embedding_model: string;
  chunk_size: number;
  chunk_overlap: number;
  created_at: string;
  updated_at: string;
}

export interface KBCreate {
  name: string;
  description: string;
  chunk_size?: number;
  chunk_overlap?: number;
}

export interface DocumentResponse {
  id: string;
  kb_id: string;
  filename: string;
  file_size: number;
  status: string;
  chunk_count: number;
  created_at: string;
}

export interface SearchResult {
  chunk_id: string;
  doc_id?: string;
  text: string;
  score: number;
  source: string;
  retrieval_source?: string;
  metadata?: Record<string, unknown>;
}

export interface CitationItem {
  index: number;
  chunk_id: string;
  doc_id: string;
  source: string;
  quote: string;
  score: number;
}

export interface QueryResponse {
  answer: string;
  model: string;
  token_count: number;
  refused: boolean;
  refusal_reason?: string | null;
  confidence: number;
  citations: CitationItem[];
  sources: SearchResult[];
}

export interface GraphSubgraph {
  nodes: Array<{
    id: string;
    label: string;
    entity_type: string;
    chunk_count: number;
  }>;
  edges: Array<{
    source: string;
    target: string;
    relation: string;
    weight: number;
  }>;
}

export interface CanvasNode {
  id: string;
  name: string;
  type: string;
  config: Record<string, any>;
  position: { x: number; y: number };
}

export interface CanvasEdge {
  source: string;
  target: string;
  label: string;
}

export interface AgentCanvasModel {
  id: string;
  name: string;
  description: string;
  nodes: CanvasNode[];
  edges: CanvasEdge[];
}

export interface CanvasRunResponse {
  canvas_id: string;
  query: string;
  final_answer: string;
  citations: CitationItem[];
  refused: boolean;
  variables: Record<string, any>;
  trace: Array<{
    step: number;
    node_id: string;
    node_name: string;
    node_type: string;
    elapsed_ms: number;
  }>;
}

export interface EvalModeReport {
  mode: string;
  top_k: number;
  total_samples: number;
  metrics: {
    recall_at_k: number;
    mrr_at_k: number;
    ndcg_at_k: number;
    citation_precision: number;
    keyword_coverage: number;
    refusal_accuracy: number;
    avg_latency_ms: number;
  };
  details: Array<{
    id: string;
    category: string;
    query: string;
    should_refuse: boolean;
    refused: boolean;
    refusal_correct: boolean;
    recall_at_k: number;
    mrr_at_k: number;
    ndcg_at_k: number;
    citation_precision: number;
    keyword_coverage: number;
    latency_ms: number;
    answer_preview: string;
    citations_count: number;
  }>;
}

export const api = {
  login: (data: LoginRequest) =>
    client.post<TokenResponse>("/auth/login", data),

  register: (data: LoginRequest & { email: string }) =>
    client.post<TokenResponse>("/auth/register", data),

  listKB: () => client.get<KBResponse[]>("/knowledge"),

  createKB: (data: KBCreate) =>
    client.post<KBResponse>("/knowledge", data),

  getKB: (kbId: string) =>
    client.get<KBResponse>(`/knowledge/${kbId}`),

  deleteKB: (kbId: string) =>
    client.delete(`/knowledge/${kbId}`),

  listDocuments: (kbId: string) =>
    client.get<DocumentResponse[]>(`/knowledge/${kbId}/documents`),

  uploadDocument: (kbId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return client.post<DocumentResponse>(
      `/knowledge/${kbId}/documents`,
      form,
      { headers: { "Content-Type": "multipart/form-data" } }
    );
  },

  ingestText: (kbId: string, filename: string, content: string) =>
    client.post<DocumentResponse>(`/knowledge/${kbId}/texts`, {
      filename,
      content,
    }),

  search: (kbId: string, query: string, topK: number = 5, mode: string = "hybrid", useGraph: boolean = true) =>
    client.post<{ results: SearchResult[] }>("/search", {
      query,
      top_k: topK,
      mode,
      use_graph: useGraph,
      kb_id: kbId,
    }),

  query: (query: string, topK: number = 5, mode: string = "hybrid", useGraph: boolean = true) =>
    client.post<QueryResponse>("/query", {
      query,
      top_k: topK,
      mode,
      use_graph: useGraph,
    }),

  getGraph: (query?: string, limit: number = 40) =>
    client.get<GraphSubgraph>("/graph", { params: { query, limit } }),

  listCanvases: () => client.get<AgentCanvasModel[]>("/agents"),

  saveCanvas: (canvas: AgentCanvasModel) =>
    client.post<AgentCanvasModel>("/agents", canvas),

  runCanvas: (canvasId: string, query: string) =>
    client.post<CanvasRunResponse>(`/agents/${canvasId}/run`, { query }),

  runEvaluation: (modes: string[] = ["dense", "sparse", "hybrid", "graph_enhanced"], topK: number = 3) =>
    client.post<{
      benchmark_name: string;
      document_count: number;
      sample_count: number;
      top_k: number;
      modes: EvalModeReport[];
      markdown_table: string;
    }>("/eval/run", { modes, top_k: topK }),
};

export default client;
