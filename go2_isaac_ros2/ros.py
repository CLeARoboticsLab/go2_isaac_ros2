import rclpy
from rclpy.node import Node
import threading
import torch

from unitree_go.msg import LowCmd, LowState
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


class Go2PubNode(Node):
    def __init__(self):
        super().__init__("go2_pub_node")

        self.low_state_pub = self.create_publisher(LowState, "/lowstate", 10)

    def publish(self, obs: dict):
        self._pub_low_state(obs)

    def _pub_low_state(self, obs: dict):
        msg = LowState()
        for i in range(12):
            msg.motor_state[i].q = obs["obs"]["joint_pos"][0, i].item()
            msg.motor_state[i].dq = obs["obs"]["joint_vel"][0, i].item()
        # TODO IMU
        self.low_state_pub.publish(msg)
