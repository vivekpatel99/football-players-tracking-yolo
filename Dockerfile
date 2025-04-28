# This assumes the container is running on a system with a CUDA GPU
# https://github.com/pytorch/pytorch#docker-image
FROM ultralytics/ultralytics:8.3.111

RUN apt-get update -y && \
    apt-get upgrade -y  \
    # Packages need for opencv
    && apt-get install -y curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgtk2.0-dev \
    pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# install UV
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Use Conda's Python as the system Python for uv
# ENV UV_SYSTEM_PYTHON=1
# ENV PATH="/opt/conda/bin:$PATH"

# WORKDIR /code
# COPY pyproject.toml uv.lock ./
# RUN uv sync
