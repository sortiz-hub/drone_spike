"""Step 12: PX4 Offboard velocity control test.

Run INSIDE the Docker container (with PX4 SITL + MAVROS running):
    source /opt/ros/humble/setup.bash
    cd /workspace/drone_spike
    python3 scripts/12_px4_offboard_test.py

This script:
1. Streams velocity setpoints (PX4 requires this before accepting Offboard mode)
2. Switches to Offboard mode
3. Arms the drone
4. Sends velocity commands: hover, move forward, move right, stop
5. Disarms

Watch the Gazebo window — you should see the drone take off and move.
"""

import time
import sys

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import TwistStamped
    from mavros_msgs.msg import State
    from mavros_msgs.srv import CommandBool, SetMode
except ImportError:
    print("ERROR: ROS 2 / MAVROS not available.")
    print("Run: source /opt/ros/humble/setup.bash")
    sys.exit(1)


class OffboardTest(Node):
    def __init__(self):
        super().__init__("offboard_test")

        self.state = State()
        self.create_subscription(State, "/mavros/state", self._state_cb, 10)

        self.vel_pub = self.create_publisher(
            TwistStamped, "/mavros/setpoint_velocity/cmd_vel", 10
        )

        self.arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.set_mode_client = self.create_client(SetMode, "/mavros/set_mode")

    def _state_cb(self, msg):
        self.state = msg

    def spin_and_wait(self, seconds: float):
        """Spin for a duration, processing callbacks."""
        end = time.time() + seconds
        while time.time() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

    def send_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float = 0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        msg.twist.angular.z = yaw_rate
        self.vel_pub.publish(msg)

    def send_velocity_for(self, vx, vy, vz, duration: float, yaw_rate=0.0):
        """Send velocity command at ~10Hz for a duration."""
        steps = int(duration * 10)
        for _ in range(steps):
            self.send_velocity(vx, vy, vz, yaw_rate)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.1)

    def call_service(self, client, request, name: str):
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f"Service {name} not available")
            return None
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        return future.result()


def main():
    rclpy.init()
    node = OffboardTest()

    # 1. Wait for MAVROS connection
    node.get_logger().info("Waiting for MAVROS connection...")
    while rclpy.ok() and not node.state.connected:
        rclpy.spin_once(node, timeout_sec=0.5)
    node.get_logger().info(f"Connected! Mode: {node.state.mode}, Armed: {node.state.armed}")

    # 2. Stream setpoints for 2s (PX4 requires this before Offboard)
    node.get_logger().info("Pre-streaming setpoints (2s)...")
    node.send_velocity_for(0.0, 0.0, 0.0, duration=2.0)

    # 3. Switch to OFFBOARD
    node.get_logger().info("Setting OFFBOARD mode...")
    req = SetMode.Request()
    req.custom_mode = "OFFBOARD"
    result = node.call_service(node.set_mode_client, req, "set_mode")
    if result:
        node.get_logger().info(f"Set mode result: mode_sent={result.mode_sent}")

    # Keep streaming while we arm (PX4 will reject Offboard if setpoints stop)
    node.send_velocity_for(0.0, 0.0, 0.0, duration=0.5)

    # 4. Arm
    node.get_logger().info("Arming...")
    req = CommandBool.Request()
    req.value = True
    result = node.call_service(node.arming_client, req, "arming")
    if result:
        node.get_logger().info(f"Arm result: success={result.success}")

    node.spin_and_wait(1.0)
    node.get_logger().info(f"State: mode={node.state.mode}, armed={node.state.armed}")

    # 5. Take off
    node.get_logger().info("Taking off (vz=2.0 for 3s)...")
    node.send_velocity_for(0.0, 0.0, 2.0, duration=3.0)

    # 6. Move forward
    node.get_logger().info("Moving forward (vx=2.0 for 3s)...")
    node.send_velocity_for(2.0, 0.0, 0.0, duration=3.0)

    # 7. Move right
    node.get_logger().info("Moving right (vy=2.0 for 3s)...")
    node.send_velocity_for(0.0, 2.0, 0.0, duration=3.0)

    # 8. Hover
    node.get_logger().info("Hovering (2s)...")
    node.send_velocity_for(0.0, 0.0, 0.0, duration=2.0)

    # 9. Land
    node.get_logger().info("Landing (vz=-1.0 for 5s)...")
    node.send_velocity_for(0.0, 0.0, -1.0, duration=5.0)

    # 10. Disarm
    node.get_logger().info("Disarming...")
    req = CommandBool.Request()
    req.value = False
    node.call_service(node.arming_client, req, "disarming")

    node.get_logger().info("Test complete!")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
