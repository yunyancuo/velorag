import React, { useEffect, useState } from "react";
import { Link, Outlet, useLocation, useNavigate } from "react-router-dom";
import { AUTH_EXPIRED_EVENT } from "../api/client";

const Layout: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [username, setUsername] = useState<string | null>(() =>
    localStorage.getItem("velorag_token")
      ? localStorage.getItem("velorag_username") || "Authenticated"
      : null
  );

  useEffect(() => {
    const syncAuth = () => {
      const hasToken = Boolean(localStorage.getItem("velorag_token"));
      setUsername(
        hasToken ? localStorage.getItem("velorag_username") || "Authenticated" : null
      );
    };
    window.addEventListener(AUTH_EXPIRED_EVENT, syncAuth);
    window.addEventListener("storage", syncAuth);
    return () => {
      window.removeEventListener(AUTH_EXPIRED_EVENT, syncAuth);
      window.removeEventListener("storage", syncAuth);
    };
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("velorag_token");
    localStorage.removeItem("velorag_username");
    setUsername(null);
    navigate("/login");
  };

  const isActive = (path: string) =>
    location.pathname === path || location.pathname.startsWith(path + "/");

  return (
    <div className="app">
      <nav className="navbar">
        <Link to="/" className="nav-brand">
          ⚡ VeloRAG <span style={{ fontSize: "0.75rem", opacity: 0.75, marginLeft: 6 }}>v0.3</span>
        </Link>
        <div className="nav-links">
          <Link to="/" className={location.pathname === "/" ? "active-link" : ""}>
            Knowledge Bases
          </Link>
          <Link to="/search/default" className={isActive("/search") ? "active-link" : ""}>
            Grounded QA & Graph
          </Link>
          <Link to="/canvas" className={isActive("/canvas") ? "active-link" : ""}>
            Agent Canvas
          </Link>
          <Link to="/eval" className={isActive("/eval") ? "active-link" : ""}>
            Eval Benchmark
          </Link>
          {username ? (
            <button type="button" className="btn-logout" onClick={handleLogout}>
              {username} · Sign Out
            </button>
          ) : (
            <Link to="/login" className="btn-logout">
              Sign In
            </Link>
          )}
        </div>
      </nav>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
};

export default Layout;
