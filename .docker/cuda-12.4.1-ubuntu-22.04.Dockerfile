FROM ubuntu:22.04 AS builder

# setup
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# install build dependencies
RUN apt-get update \
    # add ROS 2 Humble sources, see e.g. https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html
    && apt-get install -y \
        software-properties-common \
    && add-apt-repository universe \
    && apt-get update \ 
    && apt-get install -y \
        curl \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update \
    # install build tools (unavailable in base image and only required for builder stage)
    && apt-get install -y \
        git \
        cmake \
        python3 \
        python3-venv \
        python3-pip \
    # install minimal ROS 2 build utilities
    # remove ament_cmake_pytest on https://github.com/lbr-stack/lbr_fri_ros2_stack/issues/372
    && apt-get install -y \
        python3-colcon-common-extensions \
        ros-${ROS_DISTRO}-ament-cmake \
        ros-${ROS_DISTRO}-ament-cmake-pytest \
    && rm -rf /var/lib/apt/lists/*

# create ubuntu user
RUN groupadd --gid 1000 ubuntu \
    && useradd --uid 1000 --gid 1000 -m ubuntu

# non-root user installation stuff
USER ubuntu
WORKDIR /home/ubuntu

# setup
COPY --chown=ubuntu:ubuntu . ./roboreg
ENV PIP_NO_CACHE_DIR=1
SHELL ["/bin/bash", "-c"]

# clone the LBR-Stack and xarm source code for robot description only
RUN mkdir -p roboreg-deployment/src \
    && git clone \
        --depth 1 \
        -b $ROS_DISTRO \
        https://github.com/lbr-stack/lbr_fri_ros2_stack.git roboreg-deployment/src/lbr_fri_ros2_stack \
    && git clone \
        --depth 1 \
        -b $ROS_DISTRO \
        --recursive \
        --shallow-submodules \
        https://github.com/xArm-Developer/xarm_ros2.git roboreg-deployment/src/xarm_ros2

# create a virtual environment and install roboreg
RUN cd roboreg-deployment \
    && python3 -m venv roboreg-venv \
    && touch roboreg-venv/COLCON_IGNORE \
    && cd .. \
    && source roboreg-deployment/roboreg-venv/bin/activate \
    && pip3 install roboreg/ \
    && rm -rf /home/ubuntu/.cache/pip

# install robot description files
RUN cd roboreg-deployment \
    && source /opt/ros/${ROS_DISTRO}/setup.bash \
    && colcon build \
        --cmake-args -DBUILD_TESTING=0 \
        --packages-select \
            xarm_description \
            lbr_description \
    && rm -rf \
        roboreg-deployment/build \
        roboreg-deployment/log \
        roboreg-deployment/src \
        /home/ubuntu/.cache \
        /tmp/*

FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS runtime

# setup
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# create ubuntu user
RUN groupadd --gid 1000 ubuntu \
    && useradd --uid 1000 --gid 1000 -m ubuntu \
    && apt-get update \
    && apt-get install -y sudo \
    && echo ubuntu ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

# install runtime dependencies
RUN apt-get update \
    # add ROS 2 Humble sources, see e.g. https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html
    && apt-get install -y \
        software-properties-common \
    && add-apt-repository universe \
    && apt-get update \ 
    && apt-get install -y \
        curl \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update \
    # install minimal runtime utilities
    && apt-get install -y \
        python3 \
        python3-setuptools \
        ros-${ROS_DISTRO}-ament-index-python \
        ros-${ROS_DISTRO}-xacro \
        libgl1 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# copy roboreg-deployment from builder stage
COPY --chown=ubuntu:ubuntu --from=builder /home/ubuntu/roboreg-deployment/roboreg-venv /home/ubuntu/roboreg-deployment/roboreg-venv
COPY --chown=ubuntu:ubuntu --from=builder /home/ubuntu/roboreg-deployment/install /home/ubuntu/roboreg-deployment/install
COPY --chown=ubuntu:ubuntu --from=builder /home/ubuntu/roboreg/test/assets /home/ubuntu/sample-data

# non-root user
USER ubuntu
WORKDIR /home/ubuntu

# source ROS 2 workspace
RUN echo "source /home/ubuntu/roboreg-deployment/install/setup.bash" >> .bashrc
RUN echo "source /home/ubuntu/roboreg-deployment/roboreg-venv/bin/activate" >> .bashrc

# extend PATH (for CLI)
ENV PATH="$PATH:/home/ubuntu/roboreg-deployment/roboreg-venv/bin"

# run inside the roboreg folder (where data is located)
CMD ["/bin/bash"]
