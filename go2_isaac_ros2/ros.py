import rclpy
import threading
import torch
from std_msgs.msg import Float32MultiArray
from go2_isaac_ros2.env import set_action


def add_action_sub():
    action_sub = rclpy.create_node("action_sub")
    action_sub.create_subscription(
        Float32MultiArray,
        "/action",
        cmd_vel_cb,
        1,
    )
    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(action_sub,), daemon=True)
    thread.start()


def cmd_vel_cb(msg: Float32MultiArray):
    action = torch.tensor(msg.data).unsqueeze(0)
    set_action(action)
