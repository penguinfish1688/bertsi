FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install Python development tools
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipython \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    tensorboard \
    black \
    pylint \
    pytest \
    pytest-cov

# Create dev user
RUN useradd -m -s /bin/bash dev && \
    mkdir -p /workspace && \
    chown -R dev:dev /workspace

# Set working directory
WORKDIR /workspace


RUN apt-get update && apt-get install -y sudo
RUN echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev

# Switch to dev user
USER dev
