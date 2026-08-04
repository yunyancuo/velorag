import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import Login from "./pages/Login";
import KnowledgeBases from "./pages/KnowledgeBases";
import KBDetail from "./pages/KBDetail";
import Search from "./pages/Search";
import "./index.css";

const App: React.FC = () => {
  const token = localStorage.getItem("velorag_token");

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={token ? <Navigate to="/" /> : <Login />} />
        <Route element={<Layout />}>
          <Route path="/" element={<KnowledgeBases />} />
          <Route path="/kb/:kbId" element={<KBDetail />} />
          <Route path="/search/:kbId" element={<Search />} />
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
