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

client.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem("velorag_token");
      window.location.href = "/login";
    }
    return Promise.reject(err);
  }
);

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
  chunk_strategy: string;
  chunk_size: number;
  retrieval_mode: string;
  graph_enabled: boolean;
  document_count: number;
  created_at: string;
  updated_at: string;
}

export interface KBCreate {
  name: string;
  description: string;
  chunk_strategy?: string;
  retrieval_mode?: string;
  graph_enabled?: boolean;
}

export interface DocumentResponse {
  id: string;
  filename: string;
  file_type: string;
  file_size: number;
  status: string;
  chunk_count: number;
  created_at: string;
}

export interface SearchResult {
  chunk_id: string;
  text: string;
  score: number;
  source: string;
  metadata: Record<string, unknown>;
}

export const api = {
  login: (data: LoginRequest) =>
    client.post<TokenResponse>("/auth/login", data),

  register: (data: LoginRequest & { email: string }) =>
    client.post<TokenResponse>("/auth/register", data),

  listKB: () => client.get<KBResponse[]>("/knowledge-bases"),

  createKB: (data: KBCreate) =>
    client.post<KBResponse>("/knowledge-bases", data),

  getKB: (kbId: string) =>
    client.get<KBResponse>(`/knowledge-bases/${kbId}`),

  deleteKB: (kbId: string) =>
    client.delete(`/knowledge-bases/${kbId}`),

  listDocuments: (kbId: string) =>
    client.get<DocumentResponse[]>(`/knowledge-bases/${kbId}/documents`),

  uploadDocument: (kbId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return client.post<DocumentResponse>(
      `/knowledge-bases/${kbId}/documents`,
      form,
      { headers: { "Content-Type": "multipart/form-data" } }
    );
  },

  search: (kbId: string, query: string, topK: number = 10) =>
    client.post<{ results: SearchResult[]; elapsed_ms: number }>("/search", {
      query,
      top_k: topK,
      filters: { kb_id: kbId },
    }),
};

export default client;
