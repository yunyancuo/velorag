import React from "react";
import { Outlet, Link, useNavigate } from "react-router-dom";

const Layout: React.FC = () => {
  const navigate = useNavigate();

  const logout = () => {
    localStorage.removeItem("velorag_token");
    navigate("/login");
  };

  return (
    <div className="app">
      <nav className="navbar">
        <Link to="/" className="nav-brand">
          VeloRAG
        </Link>
        <div className="nav-links">
          <Link to="/">Knowledge Bases</Link>
          <button onClick={logout} className="btn-logout">
            Logout
          </button>
        </div>
      </nav>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
};

export default Layout;
