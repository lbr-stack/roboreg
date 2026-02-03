# Build arguments for version configuration
ARG PYTORCH_VERSION=2.6.0
ARG CUDA_VERSION=12.4

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn9-runtime

# setup
ENV DEBIAN_FRONTEND=noninteractive

# create ubuntu user only if it doesn't exist yet
RUN getent group ubuntu || groupadd --gid 1000 ubuntu \
    && id -u ubuntu >/dev/null 2>&1 || useradd --uid 1000 --gid 1000 -m ubuntu \
    # add ubuntu to sudoers: https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user
    && apt-get update \
    && apt-get install -y sudo \
    && echo ubuntu ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu \
    && rm -rf /var/lib/apt/lists/*

# install dependencies (most for displaying, can be removed if not needed)
RUN apt-get update \
    # install build tools (unavailable in base image and only required for builder stage)
    && apt-get install -y \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxfixes3 \
    libxi6 \
    libgl1 \
    libgl1-mesa-dri \
    libglx-mesa0 \
    libegl1 \
    libglib2.0-0 \
    libfontconfig1 \
    libfreetype6 \
    libdbus-1-3 \
    mesa-utils \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# non-root user installation stuff
USER ubuntu
WORKDIR /home/ubuntu

# setup
COPY --chown=ubuntu:ubuntu . ./roboreg
ENV PIP_NO_CACHE_DIR=1
SHELL ["/bin/bash", "-c"]

# extend PATH (for CLI)
ENV PATH="$PATH:/home/ubuntu/.local/bin"

# install roboreg (to use cached libraries, break system packages for Ubuntu 24.04)
RUN UBUNTU_VERSION=$(grep -oP '(?<=VERSION_ID=")[^"]+' /etc/os-release) && \
    echo "Detected Ubuntu $UBUNTU_VERSION" && \
    if [ "$UBUNTU_VERSION" = "24.04" ]; then \
        pip install --upgrade pip setuptools wheel --break-system-packages && \
        pip install roboreg/ --no-build-isolation --break-system-packages; \
    else \
        pip install --upgrade pip setuptools wheel && \
        pip install roboreg/ --no-build-isolation; \
    fi \
    && rm -rf /home/ubuntu/.cache/pip

# copy sample data
RUN mkdir sample-data \
    && cp -r roboreg/test/assets/* sample-data/

# run inside the roboreg folder (where data is located)
CMD ["/bin/bash"]
