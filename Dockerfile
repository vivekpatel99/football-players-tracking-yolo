# This assumes the container is running on a system with a CUDA GPU
# https://github.com/pytorch/pytorch#docker-image
FROM ultralytics/ultralytics:latest

RUN apt-get update -y && \
    apt-get upgrade -y  \
    # Packages need for opencv
    && apt-get install -y curl ffmpeg libsm6 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
