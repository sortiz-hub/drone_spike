#!/bin/bash
echo "========================================"
echo " drone-sim container started"
echo "========================================"
echo "ROS 2:   $(dpkg -l ros-humble-desktop 2>/dev/null | grep -o 'humble[^ ]*' | head -1 || echo 'NOT FOUND')"
echo "Gazebo:  $(gz sim --version 2>&1 | head -1 || echo 'NOT FOUND')"
echo "PX4:     $([ -d /opt/PX4-Autopilot ] && echo 'installed' || echo 'NOT FOUND')"
echo "Python:  $(python3 --version)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not available')"
echo "========================================"
if [ -f /workspace/drone_spike/pyproject.toml ]; then
    pip3 install -q -e /workspace/drone_spike 2>/dev/null
    echo "drone_intercept: installed (editable)"
else
    echo "drone_intercept: NOT FOUND (mount missing?)"
fi
echo "========================================"
echo "Ready. Use: docker exec -it drone-sim bash"
echo "========================================"

exec "$@"
