import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import Login from "./pages/Login";
import KnowledgeBases from "./pages/KnowledgeBases";
import KBDetail from "./pages/KBDetail";
import Search from "./pages/Search";
import AgentCanvasPage from "./pages/AgentCanvasPage";
import EvalDashboard from "./pages/EvalDashboard";
import "./index.css";

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route element={<Layout />}>
          <Route path="/" element={<KnowledgeBases />} />
          <Route path="/kb/:kbId" element={<KBDetail />} />
          <Route path="/search/:kbId" element={<Search />} />
          <Route path="/canvas" element={<AgentCanvasPage />} />
          <Route path="/eval" element={<EvalDashboard />} />
        </Route>
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </BrowserRouter>
  );
};

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
