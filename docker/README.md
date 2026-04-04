# PX4 + Gazebo + ROS 2 Docker Setup

## Prerequisites

1. **Docker Desktop** for Windows with WSL2 backend
2. **NVIDIA Container Toolkit** (for GPU-accelerated Gazebo rendering)
3. **VcXsrv** (X11 server for GUI forwarding)

## Setup

### Step 1: Install VcXsrv

Download from: https://sourceforge.net/projects/vcxsrv/

Launch **XLaunch** with:
- Multiple windows
- Start no client
- **Check** "Disable access control"

### Step 2: Install NVIDIA Container Toolkit

In PowerShell (admin):
```powershell
# Verify Docker sees your GPU
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

If that fails, install the NVIDIA Container Toolkit:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

### Step 3: Build the container

```bash
cd docker
docker compose build
```

This takes 15-30 minutes (downloads PX4, ROS 2 Humble, Gazebo Harmonic).

### Step 4: Start the container

```bash
docker compose up -d
docker exec -it drone-sim bash
```

### Step 5: Verify inside the container

```bash
# Check ROS 2
ros2 topic list

# Launch PX4 SITL + Gazebo
cd /opt/PX4-Autopilot
make px4_sitl gz_x500
```

You should see the Gazebo 3D window appear on your Windows desktop.

### Step 6: Test MAVLink + MAVROS (in a second terminal)

```bash
docker exec -it drone-sim bash
source /opt/ros/humble/setup.bash
ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14557
```

Then in a third terminal:
```bash
docker exec -it drone-sim bash
source /opt/ros/humble/setup.bash
ros2 topic list  # should see /mavros/* topics
```

## Quick Reference

| Command | What it does |
|---------|-------------|
| `docker compose build` | Build the image |
| `docker compose up -d` | Start container in background |
| `docker exec -it drone-sim bash` | Open a shell |
| `docker compose down` | Stop container |
| `docker compose down -v` | Stop and remove volumes |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No Gazebo GUI | Make sure VcXsrv is running with "Disable access control" checked |
| Black Gazebo window | Try `LIBGL_ALWAYS_INDIRECT=1` in docker-compose.yml |
| GPU not detected | Run `docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi` to verify |
| PX4 build fails | Run `cd /opt/PX4-Autopilot && make clean && make px4_sitl gz_x500` |
