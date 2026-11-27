import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, extractApiError } from "../api/client";

const Login: React.FC = () => {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const response =
        mode === "login"
          ? await api.login({ username: username.trim(), password })
          : await api.register({
              username: username.trim(),
              email: email.trim(),
              password,
            });
      localStorage.setItem("velorag_token", response.data.access_token);
      localStorage.setItem("velorag_username", response.data.username);
      navigate("/");
    } catch (err) {
      setError(extractApiError(err, "Authentication failed"));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <h1>VeloRAG</h1>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            minLength={3}
            required
          />
          {mode === "register" && (
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          )}
          <input
            type="password"
            placeholder="Password (min 8 chars)"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            minLength={mode === "register" ? 8 : 1}
            required
          />
          {error && <p className="error">{error}</p>}
          <button type="submit" disabled={submitting}>
            {submitting
              ? "Submitting..."
              : mode === "login"
                ? "Sign In"
                : "Create Account"}
          </button>
          <button
            type="button"
            className="btn-logout"
            style={{ width: "100%", marginTop: 10 }}
            onClick={() => {
              setMode(mode === "login" ? "register" : "login");
              setError("");
            }}
          >
            {mode === "login"
              ? "Need an account? Register"
              : "Already have an account? Sign In"}
          </button>
        </form>
      </div>
    </div>
  );
};

export default Login;
