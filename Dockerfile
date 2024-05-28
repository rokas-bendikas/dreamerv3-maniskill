# Instructions
#
# 1) Test setup:
#
#   docker run -it --rm --gpus all --privileged <base image> \
#     sh -c 'ldconfig; nvidia-smi'
#
# 2) Start training:
#
#   docker build -f dreamerv3/Dockerfile -t img . && \
#   docker run -it --rm --gpus all -v ~/logdir/docker:/logdir img \
#     sh -c 'ldconfig; sh embodied/scripts/xvfb_run.sh python dreamerv3/main.py \
#       --logdir "/logdir/{timestamp}" --configs atari --task atari_pong'
#
# 3) See results:
#
#   tensorboard --logdir ~/logdir/docker
#

# System
FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR 1
ENV PIP_ROOT_USER_ACTION=ignore
RUN apt-get update && apt-get install -y \
  ffmpeg git vim curl software-properties-common \
  libglew-dev x11-xserver-utils xvfb \
  && apt-get clean

# Workdir
RUN mkdir /app
WORKDIR /app

# Python
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.11-dev python3.11-venv && apt-get clean
RUN python3.11 -m venv ./venv --upgrade-deps
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools

# Envs
COPY embodied/scripts/install-minecraft.sh .
RUN sh install-minecraft.sh
COPY embodied/scripts/install-dmlab.sh .
RUN sh install-dmlab.sh
RUN pip install ale_py autorom[accept-rom-license]
RUN pip install procgen_mirror
RUN pip install crafter
RUN pip install dm_control
RUN pip install memory_maze
ENV MUJOCO_GL egl
ENV NUMBA_CACHE_DIR /tmp

# Agent
COPY dreamerv3/requirements.txt agent-requirements.txt
RUN pip install -r agent-requirements.txt \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
ENV XLA_PYTHON_CLIENT_MEM_FRACTION 0.8

# Embodied
COPY embodied/requirements.txt embodied-requirements.txt
RUN pip install -r embodied-requirements.txt

# Cloud
ENV GCS_RESOLVE_REFRESH_SECS=60
ENV GCS_REQUEST_CONNECTION_TIMEOUT_SECS=300
ENV GCS_METADATA_REQUEST_TIMEOUT_SECS=300
ENV GCS_READ_REQUEST_TIMEOUT_SECS=300
ENV GCS_WRITE_REQUEST_TIMEOUT_SECS=600
RUN chown 1000:root . && chmod 775 .

######################  Maniskill #####################
# Install os-level packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    htop \
    libegl1 \
    libxext6 \
    libjpeg-dev \
    libpng-dev  \
    libvulkan1 \
    rsync \
    tmux \
    unzip \
    vim \
    vulkan-utils \
    wget \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
RUN pip install --upgrade git+https://github.com/haosulab/ManiSkill.git@c6bd1cc9b044292046fd2887cc6c305466dfbb49
RUN pip install torch

# download physx GPU binary via sapien
RUN python -c "exec('import sapien.physx as physx;\ntry:\n  physx.enable_gpu()\nexcept:\n  pass;')"

# Copy nvidia vulkan icd and layers
COPY vulkan/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY vulkan/nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

#  Additional installation
RUN pip install hydra-core==1.3.2 wandb==0.16.6 moviepy==1.0.3 imageio==2.34.0

# Source
COPY . .