FROM ros:humble-ros-base-jammy

# change default shell
SHELL ["/bin/bash", "-c"]

# update and upgrade
RUN apt-get update
RUN apt-get install -y

# create a workspace
RUN mkdir -p /home/ros2_ws/src
WORKDIR /home/ros2_ws

# clone the xarm source code
RUN git clone https://github.com/xArm-Developer/xarm_ros2.git src/xarm_ros2 --recursive -b $ROS_DISTRO

# clone lbr-stack source code
RUN export FRI_CLIENT_VERSION=1.15 && \
    vcs import src --input https://raw.githubusercontent.com/lbr-stack/lbr_fri_ros2_stack/$ROS_DISTRO/lbr_fri_ros2_stack/repos-fri-${FRI_CLIENT_VERSION}.yaml

# build description packages
RUN source /opt/ros/$ROS_DISTRO/setup.bash &&  \
    colcon build --packages-select lbr_description xarm_description

# copy this to the container
COPY . /home

# install roboreg dependencies
RUN apt-get install -y python3-pip
RUN pip3 install \
    faiss-gpu \
    opencv-python \
    kinpy \
    matplotlib \
    numpy \
    open3d \
    rich \
    torch \
    torchvision \
    xacro

# source the workspace
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
RUN echo "source /home/ros2_ws/install/local_setup.bash" >> ~/.bashrc
