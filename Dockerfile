# Hosted playground: long-lived Python, not a static host.
# Secrets come from the platform env — do not COPY .env.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY taza_rag ./taza_rag
RUN pip install --no-cache-dir .[openai]

ENV HOST=0.0.0.0
ENV PORT=8765
EXPOSE 8765

CMD ["sh", "-c", "exec taza-rag ui --host 0.0.0.0 --port ${PORT}"]
