import React from "react";
import { Outlet, Link, useLocation } from "react-router-dom";

const Layout: React.FC = () => {
  const location = useLocation();

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
        </div>
      </nav>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
};

export default Layout;
