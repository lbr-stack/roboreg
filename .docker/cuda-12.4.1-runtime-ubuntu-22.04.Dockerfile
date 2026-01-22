FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS builder

# create ubuntu user
RUN groupadd --gid 1000 ubuntu \
    && useradd --uid 1000 --gid 1000 -m ubuntu

# add ROS 2 Jazzy sources, see e.g. https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debians.html
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
RUN apt-get update && \
    apt-get install software-properties-common -y && \
    add-apt-repository universe &&\
    apt-get update && apt-get install curl -y && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update

# install build tools (unavailable in base image and only required for builder stage)
RUN apt-get install \
        git \
        cmake \
        python3 \
        python3-venv \
        python3-pip -y

# install minimal ROS 2 build utilities
# remove ament_cmake_pytest on https://github.com/lbr-stack/lbr_fri_ros2_stack/issues/372
RUN apt-get install \
        python3-colcon-common-extensions \
        ros-${ROS_DISTRO}-ament-cmake \
        ros-${ROS_DISTRO}-ament-cmake-pytest -y

# clone the LBR-Stack and xarm source code for robot description only
WORKDIR /home/ubuntu
RUN mkdir -p roboreg-deployment/src && \
    cd roboreg-deployment/src && \
    git clone https://github.com/lbr-stack/lbr_fri_ros2_stack.git -b $ROS_DISTRO && \
    git clone https://github.com/xArm-Developer/xarm_ros2.git --recursive -b $ROS_DISTRO

# copy roboreg for installation (this is done as root)
COPY . ./roboreg

# change permissions for install
RUN chmod -R 777 \
        roboreg-deployment \
        roboreg

# non-root user installation stuff
USER ubuntu

# create a virtual environment
RUN cd roboreg-deployment && \
    python3 -m venv roboreg-venv && \
    touch roboreg-venv/COLCON_IGNORE

# change default shell
SHELL ["/bin/bash", "-c"]

# install roboreg into the venv
RUN source roboreg-deployment/roboreg-venv/bin/activate && \
    pip3 install roboreg/

# install robot description files (xarm dependencies little intertwined, require some manual installation, done above)
RUN cd roboreg-deployment && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build \
        --cmake-args -DBUILD_TESTING=0 \
        --packages-select \
            xarm_description \
            lbr_description

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# create ubuntu user
RUN groupadd --gid 1000 ubuntu \
    && useradd --uid 1000 --gid 1000 -m ubuntu \
    && apt-get update \
    && apt-get install -y sudo \
    && echo ubuntu ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

# add ROS 2 Jazzy sources, see e.g. https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debians.html
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
RUN apt-get update && \
    apt-get install software-properties-common -y && \
    add-apt-repository universe &&\
    apt-get update && apt-get install curl -y && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update

# install minimal runtime utilities
RUN apt-get install \
        python3 \
        python3-setuptools \
        ros-${ROS_DISTRO}-ament-index-python \
        ros-${ROS_DISTRO}-xacro \
        libgl1 -y

# change default shell (for ROS sourcing)
SHELL ["/bin/bash", "-c"]

# non-root user
USER ubuntu
WORKDIR /home/ubuntu

# copy roboreg-deployment from builder stage
COPY --from=builder /home/ubuntu/roboreg-deployment/roboreg-venv /home/ubuntu/roboreg-deployment/roboreg-venv
COPY --from=builder /home/ubuntu/roboreg-deployment/install /home/ubuntu/roboreg-deployment/install
COPY --from=builder /home/ubuntu/roboreg/test/assets /home/ubuntu/sample-data

# source ROS 2 workspace
RUN echo "source /home/ubuntu/roboreg-deployment/install/setup.bash" >> /home/ubuntu/.bashrc
RUN echo "source /home/ubuntu/roboreg-deployment/roboreg-venv/bin/activate" >> /home/ubuntu/.bashrc

# extend PATH (for CLI)
ENV PATH="$PATH:/home/ubuntu/roboreg-deployment/roboreg-venv/bin"

# run inside the roboreg folder (where data is located)
CMD ["/bin/bash"]
