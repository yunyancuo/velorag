"""安全审计与多租户隔离端到端回归测试。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from velorag.agent import AgentCanvas, CodeComponent
from velorag.auth.jwt import (
    create_access_token,
    decode_access_token,
    generate_api_key,
    hash_password,
    verify_password,
)
from velorag.core.database import init_db, reset_engine
from velorag.core.settings import (
    DEFAULT_DEV_SECRET_KEY,
    AuthSettings,
    Settings,
    configure_settings,
    get_settings,
)
from velorag.db.base import get_session
from velorag.db.models import ApiKey, KnowledgeBase, User
from velorag.knowledge.service import KnowledgeBaseService
from velorag.server.app import app
from velorag.server.state import reset_default_pipeline


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """为每个安全与端到端用例提供独立 SQLite 数据库、上传目录和干净管线。"""
    import velorag.knowledge.routes as kb_routes

    old_settings = get_settings()
    old_upload_dir = kb_routes.UPLOAD_DIR

    db_path = tmp_path / "test_velorag.db"
    upload_dir = tmp_path / "uploads"
    ingest_root = tmp_path / "ingest_root"
    upload_dir.mkdir(parents=True, exist_ok=True)
    ingest_root.mkdir(parents=True, exist_ok=True)

    settings = Settings()
    settings.db.url = f"sqlite:///{db_path}"
    settings.auth.secret_key = "test-strong-secret-key-for-velorag-ci-2026"
    settings.server.max_upload_mb = 1
    settings.server.allowed_ingest_root = str(ingest_root)
    configure_settings(settings)
    reset_engine()
    init_db()

    monkeypatch.setattr(kb_routes, "UPLOAD_DIR", upload_dir)
    reset_default_pipeline(seed_builtin=True)

    yield {"tmp_path": tmp_path, "upload_dir": upload_dir, "ingest_root": ingest_root}

    reset_engine()
    configure_settings(old_settings)
    monkeypatch.setattr(kb_routes, "UPLOAD_DIR", old_upload_dir)
    reset_default_pipeline(seed_builtin=True)


def _register_user(client: TestClient, username: str, email: str, password: str = "StrongPass123!") -> dict:
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": email, "password": password},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_password_hash_and_long_password():
    short_hash = hash_password("SecurePass123!")
    assert verify_password("SecurePass123!", short_hash) is True
    assert verify_password("WrongPass123!", short_hash) is False

    # 覆盖超过 bcrypt 72 字节限制的长密码场景
    long_pw = "A" * 100 + "!9z"
    long_hash = hash_password(long_pw)
    assert verify_password(long_pw, long_hash) is True
    assert verify_password("A" * 100 + "!9y", long_hash) is False
    # 畸形哈希必须安全返回 False 而不是抛异常
    assert verify_password("anything", "not-a-valid-hash") is False


def test_jwt_algorithm_enforcement_and_tampering(monkeypatch: pytest.MonkeyPatch, isolated_env):
    token = create_access_token("user-42")
    assert decode_access_token(token) == "user-42"

    # 过期令牌应被拒绝
    expired = create_access_token("user-42", expires_delta=timedelta(seconds=-5))
    assert decode_access_token(expired) is None

    # 非 access 类型令牌应被拒绝
    settings = get_settings()
    wrong_type = jwt.encode(
        {
            "sub": "user-42",
            "exp": int((datetime.now(timezone.utc) + timedelta(minutes=5)).timestamp()),
            "type": "refresh",
        },
        settings.auth.secret_key,
        algorithm="HS256",
    )
    assert decode_access_token(wrong_type) is None

    # 手工构造伪造的 alg=none 攻击 JWT，必须被拦截
    import base64
    import json

    def _b64url(data: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode("utf-8")).rstrip(b"=").decode("ascii")

    none_header = _b64url({"alg": "none", "typ": "JWT"})
    none_payload = _b64url(
        {
            "sub": "user-42",
            "exp": int((datetime.now(timezone.utc) + timedelta(minutes=5)).timestamp()),
            "type": "access",
        }
    )
    none_token = f"{none_header}.{none_payload}."
    assert decode_access_token(none_token) is None

    # 配置非法算法与生产环境弱默认密钥均必须拒绝启动
    with pytest.raises(ValueError, match="不支持的 JWT 签名算法"):
        AuthSettings(secret_key="a-very-strong-custom-secret-key", algorithm="none")

    with monkeypatch.context() as mp:
        mp.setenv("VELORAG_ENV", "production")
        with pytest.raises(ValueError, match="生产环境禁止使用默认弱 JWT 密钥"):
            AuthSettings(secret_key=DEFAULT_DEV_SECRET_KEY)


def test_api_key_lifecycle_expiry_and_disabled_user(isolated_env):
    with TestClient(app) as client:
        alice = _register_user(client, "alice_key", "alice_key@example.com")
        headers = {"Authorization": f"Bearer {alice['access_token']}"}

        # 创建 API Key 并校验前缀与鉴权
        resp = client.post("/api/v1/auth/api-keys", json={"name": "ci-key"}, headers=headers)
        assert resp.status_code == 200
        raw_key = resp.json()["raw_key"]
        key_id = resp.json()["id"]
        assert raw_key.startswith("vr-")

        me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {raw_key}"})
        assert me_resp.status_code == 200
        assert me_resp.json()["username"] == "alice_key"

        # 模拟 SQLite naive datetime 过期 API Key，不应触发 TypeError 且应拒绝鉴权
        session = get_session()
        try:
            db_key = session.query(ApiKey).filter(ApiKey.id == key_id).first()
            assert db_key is not None
            assert db_key.last_used_at is not None
            db_key.expires_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=1)
            session.commit()
        finally:
            session.close()

        expired_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {raw_key}"})
        assert expired_resp.status_code == 401

        # 恢复未过期但禁用用户：禁用账户的 API Key 也必须拒绝访问
        raw_key2, key_hash2, prefix2 = generate_api_key()
        session = get_session()
        try:
            user = session.query(User).filter(User.id == alice["user_id"]).first()
            assert user is not None
            user.is_active = False
            session.add(
                ApiKey(
                    user_id=user.id,
                    name="disabled-user-key",
                    key_hash=key_hash2,
                    key_prefix=prefix2,
                )
            )
            session.commit()
        finally:
            session.close()

        disabled_resp = client.get(
            "/api/v1/auth/me", headers={"Authorization": f"Bearer {raw_key2}"}
        )
        assert disabled_resp.status_code == 401


def test_multitenant_kb_isolation_end_to_end(isolated_env):
    with TestClient(app) as client:
        alice = _register_user(client, "alice_mt", "alice_mt@example.com")
        bob = _register_user(client, "bob_mt", "bob_mt@example.com")
        alice_headers = {"Authorization": f"Bearer {alice['access_token']}"}
        bob_headers = {"Authorization": f"Bearer {bob['access_token']}"}

        # Alice 创建私有知识库（登录用户默认私有）
        kb_resp = client.post(
            "/api/v1/knowledge",
            json={"name": "Alice Private Lab", "description": "Confidential project"},
            headers=alice_headers,
        )
        assert kb_resp.status_code == 200
        kb_data = kb_resp.json()
        kb_id = kb_data["id"]
        assert kb_data["is_public"] is False
        assert kb_data["owner_id"] == alice["user_id"]

        # Alice 通过文件上传接口写入机密文档，验证 kb_id 元数据正确注入
        secret_content = (
            "Project NebulaX uses QuantumVault protocol with secret code SECRET_ALPHA_PROJECT_999. "
            "NebulaX depends on QuantumVault for encryption."
        )
        up_resp = client.post(
            f"/api/v1/knowledge/{kb_id}/documents",
            files={"file": ("secret_plan.md", secret_content.encode("utf-8"), "text/markdown")},
            headers=alice_headers,
        )
        assert up_resp.status_code == 200
        assert up_resp.json()["status"] == "indexed"
        assert up_resp.json()["chunk_count"] >= 1

        # 列表隔离：Alice 可见，Bob 与匿名不可见
        alice_kbs = [k["id"] for k in client.get("/api/v1/knowledge", headers=alice_headers).json()]
        bob_kbs = [k["id"] for k in client.get("/api/v1/knowledge", headers=bob_headers).json()]
        anon_kbs = [k["id"] for k in client.get("/api/v1/knowledge").json()]
        assert kb_id in alice_kbs
        assert kb_id not in bob_kbs
        assert kb_id not in anon_kbs

        # 显式指定私有 kb_id 时，Bob 和匿名用户在全部入口均应被 403 拦截
        for bad_headers in (bob_headers, {}):
            assert client.get(f"/api/v1/knowledge/{kb_id}", headers=bad_headers).status_code == 403
            assert (
                client.put(
                    f"/api/v1/knowledge/{kb_id}", json={"name": "hacked"}, headers=bad_headers
                ).status_code
                == 403
            )
            assert client.delete(f"/api/v1/knowledge/{kb_id}", headers=bad_headers).status_code == 403
            assert (
                client.get(f"/api/v1/knowledge/{kb_id}/documents", headers=bad_headers).status_code
                == 403
            )
            assert (
                client.post(
                    f"/api/v1/knowledge/{kb_id}/texts",
                    json={"filename": "hack.txt", "content": "poison"},
                    headers=bad_headers,
                ).status_code
                == 403
            )
            assert (
                client.post(
                    "/api/v1/search",
                    json={"query": "SECRET_ALPHA_PROJECT_999", "kb_id": kb_id, "mode": "graph_enhanced"},
                    headers=bad_headers,
                ).status_code
                == 403
            )
            assert (
                client.post(
                    "/api/v1/query",
                    json={"query": "SECRET_ALPHA_PROJECT_999", "kb_id": kb_id, "mode": "graph_enhanced"},
                    headers=bad_headers,
                ).status_code
                == 403
            )
            assert (
                client.get(
                    "/api/v1/graph",
                    params={"query": "QuantumVault", "kb_id": kb_id},
                    headers=bad_headers,
                ).status_code
                == 403
            )
            assert (
                client.post(
                    "/api/v1/chat/sessions",
                    json={"kb_id": kb_id, "name": "steal"},
                    headers=bad_headers,
                ).status_code
                == 403
            )
            assert (
                client.post(
                    "/api/v1/agents/default-grounded-rag/run",
                    json={"query": "SECRET_ALPHA_PROJECT_999", "kb_id": kb_id},
                    headers=bad_headers,
                ).status_code
                == 403
            )

        # 不传 kb_id 时，Bob 与匿名用户也绝不能通过全局检索/问答/图谱窃取 Alice 私有数据
        for bad_headers in (bob_headers, {}):
            s_res = client.post(
                "/api/v1/search",
                json={"query": "SECRET_ALPHA_PROJECT_999 QuantumVault", "mode": "graph_enhanced"},
                headers=bad_headers,
            )
            assert s_res.status_code == 200
            assert all(
                "SECRET_ALPHA_PROJECT_999" not in r["text"] for r in s_res.json()["results"]
            )

            q_res = client.post(
                "/api/v1/query",
                json={"query": "What is SECRET_ALPHA_PROJECT_999 in NebulaX?", "mode": "graph_enhanced"},
                headers=bad_headers,
            )
            assert q_res.status_code == 200
            assert "SECRET_ALPHA_PROJECT_999" not in q_res.json()["answer"]

            g_res = client.get(
                "/api/v1/graph", params={"query": "QuantumVault"}, headers=bad_headers
            )
            assert g_res.status_code == 200
            node_names = {n["name"] for n in g_res.json()["nodes"]}
            assert "QuantumVault" not in node_names
            assert "NebulaX" not in node_names

        # Alice 本人在 Search / Query / Graph / Chat / Agent Canvas 中均正常访问并严格隔离
        a_search = client.post(
            "/api/v1/search",
            json={"query": "NebulaX QuantumVault", "kb_id": kb_id, "mode": "graph_enhanced"},
            headers=alice_headers,
        )
        assert a_search.status_code == 200
        assert any("SECRET_ALPHA_PROJECT_999" in r["text"] for r in a_search.json()["results"])

        a_query = client.post(
            "/api/v1/query",
            json={"query": "What protocol does NebulaX use?", "kb_id": kb_id, "mode": "graph_enhanced"},
            headers=alice_headers,
        )
        assert a_query.status_code == 200
        assert a_query.json()["refused"] is False
        assert "QuantumVault" in a_query.json()["answer"]

        # Chat 会话绑定私有 KB 时，问答严格限定在该 KB 内（问默认库的 Slurm 问题应拒答）
        sess_resp = client.post(
            "/api/v1/chat/sessions",
            json={"kb_id": kb_id, "name": "Alice Chat"},
            headers=alice_headers,
        )
        assert sess_resp.status_code == 200
        sess_id = sess_resp.json()["id"]

        # Bob 不能偷窥或向 Alice 的 Chat 会话发消息
        assert client.get(f"/api/v1/chat/sessions/{sess_id}", headers=bob_headers).status_code == 403
        assert (
            client.post(
                f"/api/v1/chat/sessions/{sess_id}/messages",
                json={"message": "SECRET_ALPHA_PROJECT_999"},
                headers=bob_headers,
            ).status_code
            == 403
        )

        # Alice 在私有 KB 会话中查私有内容成功
        msg_ok = client.post(
            f"/api/v1/chat/sessions/{sess_id}/messages",
            json={"message": "What protocol does NebulaX use?"},
            headers=alice_headers,
        )
        assert msg_ok.status_code == 200
        assert "QuantumVault" in msg_ok.json()["content"]

        # Alice 在私有 KB 会话中查默认库内容（Slurm）不得跨库命中
        msg_cross = client.post(
            f"/api/v1/chat/sessions/{sess_id}/messages",
            json={"message": "AI开源实验室算力集群使用什么调度系统？"},
            headers=alice_headers,
        )
        assert msg_cross.status_code == 200
        assert "Slurm" not in msg_cross.json()["content"]

        # Agent Canvas 运行传私有 kb_id：Alice 命中私有内容
        canvas_run = client.post(
            "/api/v1/agents/default-grounded-rag/run",
            json={"query": "What protocol does NebulaX use?", "kb_id": kb_id},
            headers=alice_headers,
        )
        assert canvas_run.status_code == 200
        assert "QuantumVault" in canvas_run.json()["final_answer"]

        # 删除私有 KB 后，向量和图谱中不再存在残留数据
        del_resp = client.delete(f"/api/v1/knowledge/{kb_id}", headers=alice_headers)
        assert del_resp.status_code == 200
        after_search = client.post(
            "/api/v1/search",
            json={"query": "SECRET_ALPHA_PROJECT_999", "mode": "graph_enhanced"},
            headers=alice_headers,
        )
        assert all("SECRET_ALPHA_PROJECT_999" not in r["text"] for r in after_search.json()["results"])


def test_kb_service_cannot_update_others_public_kb(isolated_env):
    session = get_session()
    try:
        alice = User(username="svc_alice", email="svc_alice@example.com", hashed_password=hash_password("Pass123456!"))
        bob = User(username="svc_bob", email="svc_bob@example.com", hashed_password=hash_password("Pass123456!"))
        session.add_all([alice, bob])
        session.commit()
        session.refresh(alice)
        session.refresh(bob)

        kb = KnowledgeBaseService.create(
            session, alice, name="Public Docs", description="Original", is_public=True
        )
        assert isinstance(kb, KnowledgeBase)

        # Bob 可读公开 KB，但绝不能通过 update 修改他人公开 KB
        assert KnowledgeBaseService.get(session, kb.id, owner=bob) is not None
        assert KnowledgeBaseService.update(session, kb.id, owner=bob, name="Tampered") is None

        # Alice 本人可正常修改
        updated = KnowledgeBaseService.update(session, kb.id, owner=alice, name="Updated By Owner")
        assert updated is not None
        assert updated.name == "Updated By Owner"
    finally:
        session.close()


def test_path_traversal_upload_limits_and_ingest_guard(isolated_env):
    with TestClient(app) as client:
        user = _register_user(client, "uploader", "uploader@example.com")
        headers = {"Authorization": f"Bearer {user['access_token']}"}
        kb_id = client.post(
            "/api/v1/knowledge", json={"name": "Upload Guard KB"}, headers=headers
        ).json()["id"]

        # 路径遍历与非法扩展名拦截（上传文件与直接文本接口）
        for bad_name in ("../../evil.txt", "../secret.md", "sub/dir.txt", "payload.exe", "script.sh"):
            r_up = client.post(
                f"/api/v1/knowledge/{kb_id}/documents",
                files={"file": (bad_name, b"hello", "text/plain")},
                headers=headers,
            )
            assert r_up.status_code == 400, f"{bad_name} 应被拦截，实际状态码 {r_up.status_code}"

            r_txt = client.post(
                f"/api/v1/knowledge/{kb_id}/texts",
                json={"filename": bad_name, "content": "hello"},
                headers=headers,
            )
            assert r_txt.status_code == 400

        # 空文件应返回 400
        r_empty = client.post(
            f"/api/v1/knowledge/{kb_id}/documents",
            files={"file": ("empty.txt", b"", "text/plain")},
            headers=headers,
        )
        assert r_empty.status_code == 400

        # 超出 1MB 限制的超大文件返回 413
        oversized = b"x" * (1024 * 1024 + 128)
        r_big = client.post(
            f"/api/v1/knowledge/{kb_id}/documents",
            files={"file": ("huge.txt", oversized, "text/plain")},
            headers=headers,
        )
        assert r_big.status_code == 413

        # /ingest 端点未认证应返回 401
        assert client.post("/ingest", json={"path": "note.txt"}).status_code == 401

        # /ingest 路径穿越根目录应返回 400
        assert (
            client.post("/ingest", json={"path": "../../etc/passwd"}, headers=headers).status_code
            == 400
        )

        # 允许根目录内的合法文件正常入库
        valid_file = isolated_env["ingest_root"] / "allowed.txt"
        valid_file.write_text("Allowed local ingest document for VeloRAG.", encoding="utf-8")
        ok_ingest = client.post(
            "/ingest", json={"path": "allowed.txt", "kb_id": kb_id}, headers=headers
        )
        assert ok_ingest.status_code == 200
        assert ok_ingest.json()["ingested_chunks"] >= 1


def test_code_component_blocks_rce_and_allows_safe_transforms(isolated_env):
    # 1. 合法安全变换应正常执行
    canvas = AgentCanvas("Safe Code Canvas")
    canvas.add_component(
        CodeComponent(
            "code_ok",
            "Safe Transform",
            code="items = [1, 2, 3]\noutput = {'total': sum(items), 'query_len': len(query)}",
        )
    )
    ctx = asyncio.run(canvas.run("hello"))
    assert ctx.variables["total"] == 6
    assert ctx.variables["query_len"] == 5

    # 2. 经典 RCE 沙箱逃逸与 DoS 负载必须全部被安全校验拦截
    malicious_payloads = [
        "output = {'x': ().__class__.__base__.__subclasses__()}",
        "import os\noutput = {'x': os.getcwd()}",
        "output = {'x': __import__('os').system('echo pwned')}",
        "while True:\n    pass",
        "output = {'x': open('/etc/passwd').read()}",
        "output = {'x': 'A' * 999999}",
    ]
    for payload in malicious_payloads:
        bad_canvas = AgentCanvas("Evil Canvas")
        bad_canvas.add_component(CodeComponent("evil", "Evil", code=payload))
        with pytest.raises((ValueError, RuntimeError)):
            asyncio.run(bad_canvas.run("test"))

    # 3. 通过 REST API 保存恶意 CodeComponent 画布必须返回 400
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/agents",
            json={
                "name": "Malicious",
                "nodes": [
                    {"id": "begin", "name": "Begin", "type": "begin", "config": {}},
                    {
                        "id": "code1",
                        "name": "Evil",
                        "type": "code",
                        "config": {"code": "output = {'x': ().__class__.__mro__}"},
                    },
                ],
                "edges": [{"source": "begin", "target": "code1", "label": "default"}],
            },
        )
        assert resp.status_code == 400
