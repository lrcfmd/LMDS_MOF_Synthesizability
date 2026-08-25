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
WORKDIR /app
COPY --from=builder /app/dist/*.whl /app/
# Install the CPU-only torch first so the wheel's torch==2.0.1 requirement is
# satisfied by the CPU build rather than the default multi-GB CUDA wheel.
RUN pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir /app/*.whl

# Model artifacts are NOT baked into the image — they are provided at runtime.
# The application reads them from MODEL_ARTIFACTS_DIR; mount the models there
# (e.g. `-v /path/to/models:/models`). See src/model/artifacts/README.md.
ENV MODEL_ARTIFACTS_DIR=/models
RUN mkdir -p /data/output /models
ENTRYPOINT ["model-run"]
