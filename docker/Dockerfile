# ── 阶段1：依赖构建 ───────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# 先复制 requirements，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── 阶段2：运行镜像 ───────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# 运行时系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 复制已安装的包
COPY --from=builder /install /usr/local

# 复制项目代码
COPY . .

# 创建数据目录
RUN mkdir -p /app/data/faiss_index

# 非 root 用户运行
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 预热：提前下载向量模型（构建时缓存）
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('shibing624/text2vec-base-chinese')" || true

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
