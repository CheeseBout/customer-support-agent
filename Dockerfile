FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# Dependencies first so code changes do not reinstall them.
COPY pyproject.toml ./
COPY src ./src
RUN pip install .

COPY config ./config
COPY knowledge ./knowledge

# Models, the vector index, conversations and sessions live here: mount a volume on /app/data.
RUN useradd --create-home app && mkdir -p /app/data && chown -R app /app/data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["support-agent", "serve", "--host", "0.0.0.0", "--port", "8000"]
