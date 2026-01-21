FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

# update
RUN apt-get update

# add ubuntu to sudoers: https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user
RUN apt-get install -y sudo \
    && echo ubuntu ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

# install tools (unavailable in base image)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install \
        git \
        cmake \
        python3 \
        python3-pip \
        ninja-build \
        libgl1 -y

# change default shell
SHELL ["/bin/bash", "-c"]

# add ROS 2 Jazzy sources, see e.g. https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debians.html
ENV ROS_DISTRO=jazzy
RUN apt-get install software-properties-common -y && \
    add-apt-repository universe &&\
    apt-get update && apt-get install curl -y && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update

# install minimal ROS 2 utilities:
#   - build: ament_cmake
#   - package registry: ament_index_python
#   - parse: xacro
# remove ament_cmake_pytest on https://github.com/lbr-stack/lbr_fri_ros2_stack/issues/372
RUN apt-get install python3-colcon-common-extensions \
        ros-${ROS_DISTRO}-ament-cmake \
        ros-${ROS_DISTRO}-ament-cmake-pytest \
        ros-${ROS_DISTRO}-ament-index-python \
        ros-${ROS_DISTRO}-xacro -y

# clone the LBR-Stack and xarm source code for robot description only
WORKDIR /home/ubuntu
RUN mkdir -p ros2_ws/src && \
    cd ros2_ws/src && \
    git clone https://github.com/lbr-stack/lbr_fri_ros2_stack.git -b $ROS_DISTRO && \
    git clone https://github.com/xArm-Developer/xarm_ros2.git --recursive -b $ROS_DISTRO

# copy roboreg for installation (this is done as root)
COPY . ./roboreg

# change permissions for install
RUN chmod -R 777 roboreg && \
    chmod -R 777 ros2_ws

# non-root user installation stuff (previously for rosdep)
USER ubuntu

# install robot description files (xarm dependencies little intertwined, require some manual installation, done above)
RUN cd ros2_ws && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build \
        --cmake-args -DBUILD_TESTING=0 \
        --packages-select \
            xarm_description \
            lbr_description && \
    rm -r src

# source ROS 2 workspace
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/ubuntu/.bashrc && \
    echo "source /workspace/ros2_ws/install/local_setup.bash" >> /home/ubuntu/.bashrc

# extend PYTHONPATH and PATH (for CLI)
ENV PYTHONPATH="/home/ubuntu/.local/lib/python3.12/site-packages"
ENV PATH="$PATH:/home/ubuntu/.local/bin"

# install roboreg
RUN pip3 install roboreg/ --break-system-packages

# limit concurrent compilation for ninja, refer https://github.com/NVlabs/nvdiffrast/issues/201
ENV MAX_JOBS=2

# run inside the roboreg folder (where data is located)
WORKDIR /home/ubuntu/roboreg
CMD ["/bin/bash"]
