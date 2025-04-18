# This assumes the container is running on a system with a CUDA GPU
# https://github.com/pytorch/pytorch#docker-image
FROM ultralytics/ultralytics:8.3.111

RUN apt-get update -y && \
    apt-get upgrade -y  \
    # Packages need for opencv
    && apt-get install -y curl ffmpeg libsm6 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# install UV
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# # Tell uv to install packages into the /usr/local prefix (system-wide within the container)
# ENV UV_PROJECT_ENVIRONMENT="/usr/local/python"

# WORKDIR /code
# COPY pyproject.toml .
# RUN uv sync