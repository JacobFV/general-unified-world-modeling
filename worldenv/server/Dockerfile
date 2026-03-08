FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "openenv-core>=0.2.0" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.24.0" \
    "pydantic>=2.0"

COPY . /app/worldenv/

ENV PYTHONPATH="/app/worldenv:/app/worldenv/server:$PYTHONPATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--app-dir", "/app/worldenv", "--host", "0.0.0.0", "--port", "8000"]
