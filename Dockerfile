# syntax=docker/dockerfile:1
FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/index.html frontend/tsconfig.json frontend/vite.config.ts ./
COPY frontend/src ./src
RUN npm run build

FROM python:3.13-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STORAGE_ROOT=/var/lib/workbench \
    HF_HOME=/var/cache/workbench/huggingface \
    FRONTEND_DIST=/app/static \
    ASR_DEVICE=cpu \
    ASR_COMPUTE_TYPE=auto
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home workbench \
    && mkdir -p /var/lib/workbench /var/cache/workbench/huggingface \
    && chown -R workbench:workbench /var/lib/workbench /var/cache/workbench
WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# 明确列举构建输入，不复制 .env、data、历史记录或本机模型。
COPY backend/app ./app
COPY --from=frontend /build/dist ./static
USER workbench
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1
# 任务状态仍在进程内，必须使用单个 worker。
CMD ["python", "-m", "uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
