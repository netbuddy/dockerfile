FROM osrf/ros:noetic-desktop-full

WORKDIR /app

# 依赖安装
COPY sources.list_ubuntu_20.04_x64 /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y \
    ninja-build exiftool ninja-build protobuf-compiler libeigen3-dev genromfs xmlstarlet libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev python3-pip gawk python3-catkin-tools unzip git wget tmux

RUN pip3 install packaging numpy empy toml pyyaml jinja2 pyargparse kconfiglib jsonschema future pyquaternion pyros-genmsg

# Gazebo安装
RUN apt-get remove -y \
    gazebo* libgazebo* ros-noetic-gazebo*

RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list' && \
    wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -

RUN apt-get update && \
    apt-get install -y \
    gazebo11 libgazebo11-dev

RUN apt-get install -y ros-noetic-moveit-msgs ros-noetic-object-recognition-msgs ros-noetic-octomap-msgs ros-noetic-camera-info-manager  ros-noetic-control-toolbox ros-noetic-polled-camera ros-noetic-controller-manager ros-noetic-transmission-interface ros-noetic-joint-limits-interface ros-noetic-mavros ros-noetic-mavros-extras

RUN git clone https://gitee.com/robin_shaun/XTDrone.git && \
    cd XTDrone && \
    git checkout 1_13_2 && \
    git submodule update --init --recursive &&

RUN mkdir -p catkin_ws/{src,scripts} && \
    . /opt/ros/noetic/setup.sh && \
    cd catkin_ws && \
    cp -r ../XTDrone/sitl_config/gazebo_ros_pkgs src/ && \
    catkin init && \
    catkin build &&

RUN apt-get install -y git-lfs && \
    git lfs install

RUN git clone https://e.coding.net/g-ywlk6881/oslab/xtdrone.git && \
    cd xtdrone && \
    mkdir ~/.gazebo && \
    unzip -d ~/.gazebo models.zip && \
    unzip -d /usr/share GeographicLib.zip && \
    cd .. && \
    rm -rf xtdrone

# PX4配置
RUN git clone https://ghp.ci/https://github.com/PX4/PX4-Autopilot.git && \
    mv PX4-Autopilot PX4_Firmware && \
    cd PX4_Firmware && \
    git checkout -b xtdrone/dev v1.13.2 && \
    sed -i 's/https:\/\/github\.com/https:\/\/ghp\.ci\/&/' .gitmodules && \
    git submodule sync && \
    git submodule update --init --recursive && \
    make px4_sitl_default gazebo &&

RUN cat <<EOL > ~/.bashrc
source /app/catkin_ws/devel/setup.bash
source /app/PX4_Firmware/Tools/setup_gazebo.bash /app/PX4_Firmware/ /app/PX4_Firmware/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/app/PX4_Firmware
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/app/PX4_Firmware/Tools/sitl_gazebo
EOL

# XTDrone配置
RUN cd XTDrone && \
## 修改启动脚本文件
    cp sitl_config/init.d-posix/* ../PX4_Firmware/ROMFS/px4fmu_common/init.d-posix/ && \
## 添加launch文件
    cp -r sitl_config/launch/* ../PX4_Firmware/launch/ && \
## 添加世界文件
    cp sitl_config/worlds/* ../PX4_Firmware/Tools/sitl_gazebo/worlds/ && \
## 修改部分插件
    cp sitl_config/gazebo_plugin/gimbal_controller/gazebo_gimbal_controller_plugin.cpp ../PX4_Firmware/Tools/sitl_gazebo/src && \
    cp sitl_config/gazebo_plugin/gimbal_controller/gazebo_gimbal_controller_plugin.hh ../PX4_Firmware/Tools/sitl_gazebo/include && \
    cp sitl_config/gazebo_plugin/wind_plugin/gazebo_ros_wind_plugin_xtdrone.cpp ../PX4_Firmware/Tools/sitl_gazebo/src && \
    cp sitl_config/gazebo_plugin/wind_plugin/gazebo_ros_wind_plugin_xtdrone.h ../PX4_Firmware/Tools/sitl_gazebo/include && \
## 修改CMakeLists.txt
    cp sitl_config/CMakeLists.txt ../PX4_Firmware/Tools/sitl_gazebo && \
## 修改部分模型文件
    cp -r sitl_config/models/* ../PX4_Firmware/Tools/sitl_gazebo/models/ && \
## 替换同名文件
    cd ~/.gazebo/models/ && \
    rm -rf stereo_camera/ 3d_lidar/ 3d_gpu_lidar/ hokuyo_lidar/ && \
    cd /app/PX4_Firmware && \
    rm -rf build/ && \
    . /opt/ros/noetic/setup.sh && \
    make px4_sitl_default gazebo