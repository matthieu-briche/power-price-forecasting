FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so code changes don't invalidate this layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

COPY scripts ./scripts

# Run as a non-root user.
RUN useradd --create-home appuser && mkdir -p /app/reports && chown -R appuser /app
USER appuser

ENTRYPOINT ["powerfc"]
CMD ["backtest", "--source", "synthetic", "--out-dir", "/app/reports"]
