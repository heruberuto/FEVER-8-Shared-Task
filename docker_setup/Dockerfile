# Start from NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Prevent interactive prompts during build
ARG DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
ENV PATH=${CUDA_HOME}/bin:${PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    unzip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create ubuntu user and set up home directory
RUN useradd -m -d /home/ubuntu -s /bin/bash ubuntu && \
    chown -R ubuntu:ubuntu /home/ubuntu

# Switch to the ubuntu user
USER ubuntu
WORKDIR /home/ubuntu

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /home/ubuntu/miniconda3 && \
    rm ~/miniconda.sh

# Add conda to path
ENV PATH=/home/ubuntu/miniconda3/bin:${PATH}

# Initialize conda for the ubuntu user
RUN conda init bash && \
    echo "conda activate base" >> ~/.bashrc

# Copy environment.yml
COPY --chown=ubuntu:ubuntu environment.yml /home/ubuntu/

# Create conda environment from yml
RUN conda env update -n base -f /home/ubuntu/environment.yml && \
    conda clean -afy

# Copy of Averitec Baseline
COPY --chown=ubuntu:ubuntu AVeriTeC-Shared-Task-2025/ /home/ubuntu/AVeriTeC-Shared-Task-2025/

# Make any scripts executable
RUN chmod +x /home/ubuntu/AVeriTeC-Shared-Task-2025/*.sh

# Verify CUDA installation
RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

CMD ["/bin/bash"]