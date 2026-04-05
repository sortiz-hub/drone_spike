FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# -------------------------------------------------------
# 1. Base dependencies
# -------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    gnupg2 \
    lsb-release \
    software-properties-common \
    sudo \
    wget \
    python3 \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    x11-apps \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------
# 2. ROS 2 Humble
# -------------------------------------------------------
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | \
    gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    ros-humble-mavros \
    ros-humble-mavros-extras \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Install MAVROS geographiclib datasets
RUN wget -q https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh -O /tmp/install_geo.sh && \
    bash /tmp/install_geo.sh && rm /tmp/install_geo.sh

# -------------------------------------------------------
# 3. Gazebo Harmonic
# -------------------------------------------------------
RUN curl -sSL https://packages.osrfoundation.org/gazebo.gpg | \
    gpg --dearmor -o /usr/share/keyrings/gazebo-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gazebo-archive-keyring.gpg] \
    http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" > /etc/apt/sources.list.d/gazebo-stable.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    gz-harmonic \
    ros-humble-ros-gzharmonic \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------
# 4a. PX4 Autopilot — shallow clone (cacheable layer)
# -------------------------------------------------------
RUN git clone --depth 1 --branch v1.15.2 \
    https://github.com/PX4/PX4-Autopilot.git /opt/PX4-Autopilot

# -------------------------------------------------------
# 4b. PX4 submodules — only the ones needed for SITL + Gazebo
# -------------------------------------------------------
RUN cd /opt/PX4-Autopilot && \
    git submodule update --init --depth 1 --jobs 8 \
        src/lib/matrix \
        src/lib/crypto/libtommath \
        src/lib/crypto/libtomcrypt \
        src/drivers/uavcan/libuavcan \
        src/modules/mavlink/mavlink \
        Tools/simulation/gz \
        Tools/jMAVSim \
        platforms/common/uORB/Subscription \
    && git submodule update --init --depth 1 --jobs 8 \
        mavlink/include/mavlink/v2.0 || true

# -------------------------------------------------------
# 4c. PX4 build deps + SITL build (separate layer for faster retries)
# -------------------------------------------------------
RUN cd /opt/PX4-Autopilot && \
    bash Tools/setup/ubuntu.sh --no-nuttx && \
    sed -i "s/nuttx_git_tags)\[-1\]/nuttx_git_tags)[-1] if nuttx_git_tags else 'v0.0.0'/" \
        src/lib/version/px_update_git_header.py && \
    make px4_sitl_default || true

# -------------------------------------------------------
# 5. Python deps for drone_intercept
# -------------------------------------------------------
RUN pip3 install --no-cache-dir gymnasium stable-baselines3 numpy matplotlib torch opencv-python-headless PyFlyt --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir --upgrade setuptools pip numpy

# -------------------------------------------------------
# 6. Workspace + shell setup (keep last to avoid cache busts)
# -------------------------------------------------------
RUN mkdir -p /workspace
WORKDIR /workspace

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "export GZ_SIM_RESOURCE_PATH=/opt/PX4-Autopilot/Tools/simulation/gz/models:/opt/PX4-Autopilot/Tools/simulation/gz/worlds" >> /root/.bashrc

COPY docker/entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

ENTRYPOINT ["/root/entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]
