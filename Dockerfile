# API + web page for the UK Company Failure Predictor.
# The trained model (models/) is mounted at run time by docker-compose.yml; the Companies House
# API key comes from the environment. Neither is baked into the image.
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv

WORKDIR /srv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/srv/.venv \
    PATH="/srv/.venv/bin:$PATH" \
    MPLCONFIGDIR=/tmp/matplotlib \
    CFP_MODEL_DIR=/srv/models \
    CFP_FRONTEND_DIR=/srv/app \
    CFP_DATA_DIR=/srv/data

# Dependencies first so code changes do not reinstall them. Serving needs no dev or train tools.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-default-groups --no-install-project
COPY src ./src
RUN uv sync --locked --no-default-groups
COPY app ./app

RUN useradd --create-home app && mkdir -p /srv/data /srv/models && chown app /srv/data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "cfp.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
