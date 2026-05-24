# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# System setup
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git curl vim wget build-essential net-tools iputils-ping nano screen \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Agentlace deps.
RUN apt update && apt install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory.
WORKDIR /app

# Copy package.
COPY ./pyproject.toml ./pyproject.toml
COPY ./uv.lock ./uv.lock
COPY ./src ./src

# uv.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"
RUN uv --version

# Install package.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

CMD ["/bin/bash"]
