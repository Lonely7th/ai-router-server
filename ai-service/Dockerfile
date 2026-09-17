FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AI_HOST=0.0.0.0 \
    AI_PORT=8000 \
    WEB_CONCURRENCY=1

RUN addgroup --system app && adduser --system --ingroup app app
COPY --from=builder /install /usr/local

USER app
WORKDIR /app
EXPOSE 8000

CMD ["sh", "-c", "uvicorn jj_office_ai.main:app --host ${AI_HOST} --port ${AI_PORT} --workers ${WEB_CONCURRENCY}"]
