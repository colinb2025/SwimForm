# SwimForm web app.
#   docker build -t swimform .                       # synthetic pose (demo data), small image
#   docker build -t swimform --build-arg POSE=1 .    # + ultralytics/torch for the real model
# See docs/SETUP.md for the environment variables.
FROM python:3.12-slim

ARG POSE=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    SWIMFORM_DATA=/data SWIMFORM_SECURE_COOKIES=1

# OpenCV needs these shared libraries even headless.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# Editable install: the package finds rules/ relative to the source tree.
RUN pip install -e ".[app,postgres,ses]" && if [ "$POSE" = "1" ]; then pip install -e ".[pose]"; fi
COPY rules ./rules
COPY app ./app

RUN useradd --create-home --uid 1000 swimform && mkdir -p /data && chown swimform /data
USER swimform
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status == 200 else 1)"

# One process: background analysis jobs run in this process (see app/jobs.py).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
