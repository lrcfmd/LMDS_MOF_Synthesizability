# syntax=docker/dockerfile:1
# 3.11 is required: the LOF/LUNAR/scaler pickles were saved with scikit-learn
# 1.2.0, which has no cp312 wheels. See pyproject.toml for the full rationale.
ARG PYTHON_VERSION=3.11

# Stage 1: Build wheel using uv
FROM python:${PYTHON_VERSION}-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv build --wheel

# Stage 2: Minimal runtime image
FROM python:${PYTHON_VERSION}-slim
RUN groupadd -r lmds && useradd -r -g lmds -u 1000 -m -d /home/lmds lmds
WORKDIR /app
COPY --from=builder /app/dist/*.whl /app/
# Install the CPU-only torch first so the wheel's torch==2.0.1 requirement is
# satisfied by the CPU build rather than the default multi-GB CUDA wheel.
RUN pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir /app/*.whl && rm -f /app/*.whl

# Model artifacts are NOT baked into the image — they are declared as `assets`
# in schema/model.json and downloaded at run time by the runner to their paths
# under the WORKDIR (./artifacts/…). The platform runs the container as uid
# 1000; the runner also creates /data/input (file inputs) and /data/output
# (results), so /app and all of /data (and /home/lmds) must be writable by it.
RUN mkdir -p /data/input /data/output && chown -R 1000:1000 /app /data /home/lmds
USER 1000:1000
ENTRYPOINT ["model-run"]
