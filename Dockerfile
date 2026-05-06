FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0+PTX"
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/I-Scene-project

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ffmpeg \
        git \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        ninja-build \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3-pip \
        wget && \
    python3.10 -m venv "${VIRTUAL_ENV}" && \
    python -m pip install --upgrade pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

# Install the tested dependency set from the repository. This file currently
# pins Python 3.10 / Torch 2.4.0 / CUDA 12.1 compatible binary packages.
COPY requirements.txt ./
RUN python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.0+cu121 \
        torchvision==0.19.0+cu121
RUN grep -v "autonomousvision/mip-splatting.git" requirements.txt > /tmp/requirements-without-dgr.txt && \
    python -m pip install --no-build-isolation -r /tmp/requirements-without-dgr.txt

# This extension imports torch from setup.py during metadata generation, so it
# must be installed after torch exists in the environment. Build it locally to
# avoid incompatible prebuilt CUDA binaries on newer GPUs.
RUN python -m pip install --no-build-isolation --no-deps --force-reinstall \
        "git+https://github.com/autonomousvision/mip-splatting.git@dda02ab5ecf45d6edb8c540d9bb65c7e451345a9#subdirectory=submodules/diff-gaussian-rasterization"

# The nvdiffrast wheel in requirements.txt is convenient for some machines, but
# it can be incompatible with newer GPU architectures such as H100. Reinstall
# from NVlabs source so the plugin is built/JIT-compiled against this CUDA 12.1
# devel image instead of relying on a prebuilt shared object.
ARG NVDIFFRAST_REF=v0.3.3
RUN python -m pip uninstall -y nvdiffrast && \
    python -m pip install --no-deps --force-reinstall \
        "git+https://github.com/NVlabs/nvdiffrast.git@${NVDIFFRAST_REF}"

CMD ["/bin/bash"]
