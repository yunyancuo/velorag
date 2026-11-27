FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 安装编译扩展所需的最小系统依赖，并在同层清理缓存
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 优先拷贝项目元数据与依赖清单，充分利用 Docker 构建层缓存
COPY pyproject.toml README.md ./
RUN mkdir -p src/velorag && touch src/velorag/__init__.py \
    && pip install --no-cache-dir ".[full]" \
    && rm -rf src/velorag

# 拷贝业务源码并以无重复下载依赖方式完成安装
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

# 创建非 root 运行用户与持久化数据目录
RUN groupadd -r velorag && useradd -r -g velorag -m velorag \
    && mkdir -p /data/velorag /app/data \
    && chown -R velorag:velorag /app /data

USER velorag

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "velorag.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
