import rclpy
import threading
import torch

from unitree_go.msg import LowCmd
from go2_isaac_ros2.env import set_action


def add_lowcmd_sub():  # todo: this should actually sub to /lowcmd
    lowcmd_sub = rclpy.create_node("lowcmd_sub")
    lowcmd_sub.create_subscription(
        LowCmd,
        "/lowcmd",
        lowcmd_cb,
        1,
    )
    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(lowcmd_sub,), daemon=True)
    thread.start()


def lowcmd_cb(msg: LowCmd):
    action = torch.zeros(12)
    for i in range(12):
        action[i] = msg.motor_cmd[i].q
    set_action(action.unsqueeze(0))
