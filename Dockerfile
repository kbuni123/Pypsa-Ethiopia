FROM python:3.11-slim

# curl is needed by the HEALTHCHECK; build tools only for any sdist fallbacks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Every module app.py imports. The data folder is NOT baked in: it holds the
# multi-GB ERA5 cutout, so docker-compose mounts it at /app/data instead.
COPY app.py model.py data.py hydro.py timeseries.py osm_grid.py setup_data.py ./

RUN mkdir -p data results logs

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
