FROM nvcr.io/nvidia/pytorch:23.07-py3

# Build arguments
ARG USER_ID
ARG GROUP_ID
ARG USER

# Create non-root user: https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user
RUN groupadd --gid $GROUP_ID $USER \
    && useradd --uid $USER_ID --gid $GROUP_ID -m $USER \
    # Add sudo support
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER

# change default shell
SHELL ["/bin/bash", "-c"]

# update, upgrade
RUN apt-get update && \
    apt-get upgrade -y

# install ROS 2 Rolling, see e.g. https://docs.ros.org/en/rolling/Installation/Ubuntu-Install-Debians.html
ARG DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=rolling
RUN apt-get install software-properties-common -y && \
    add-apt-repository universe &&\
    apt-get update && apt-get install curl -y && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && apt-get upgrade -y && \
    apt-get install ros-${ROS_DISTRO}-ros-base -y && \
    apt-get install ros-dev-tools -y && \
    rosdep init

# install some more dependencies
# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get install -y \
    libgl1 \
    ros-${ROS_DISTRO}-xacro

# clone the LBR-Stack and xarm source code for robot description only
ENV FRI_CLIENT_VERSION=1.15
RUN mkdir -p ros2_ws/src && \
    cd ros2_ws/src && \
    git clone https://github.com/lbr-stack/lbr_fri_ros2_stack.git -b $ROS_DISTRO && \
    git clone https://github.com/xArm-Developer/xarm_ros2.git --recursive -b $ROS_DISTRO

# copy roboreg for installation (this is done as root...)
COPY . ./roboreg

# change permissions for install
RUN chmod -R 777 roboreg && \
    chmod -R 777 ros2_ws

# NON ROOT USER INSTALLATION STUFF...
USER $USER

# install robot description files (xarm dependencies little intertwined, require some manual installation)
RUN cd ros2_ws && \
    apt-get install -y \
        ros-${ROS_DISTRO}-gazebo-ros \
        ros-${ROS_DISTRO}-control-msgs \
        ros-${ROS_DISTRO}-ros2-control && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --symlink-install --packages-select \
        xarm_description \
        xarm_controller \
        xarm_msgs \
        xarm_sdk \
        xarm_api
RUN cd ros2_ws && \
    rosdep update && \
    rosdep install --from-paths \
        src/lbr_fri_ros2_stack/lbr_description \
        -i -r -y --rosdistro=${ROS_DISTRO} && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --symlink-install --packages-select \
        lbr_description

# source ROS 2 workspace
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/$USER/.bashrc && \
    echo "source /home/$USER/ros2_ws/install/local_setup.bash" >> /home/$USER/.bashrc

# extend PYTHONPATH and PATH (for CLI)
ENV PYTHONPATH="/home/$USER/.local/lib/python3.10/site-packages"
ENV PATH="$PATH:/home/$USER/.local/bin"

# install roboreg
RUN pip3 install roboreg/

# limit concurrent compilation for ninja, refer https://github.com/NVlabs/nvdiffrast/issues/201
ENV MAX_JOBS=1

# run inside the roboreg folder (where data is located)
WORKDIR /workspace/roboreg
CMD ["/bin/bash"]
