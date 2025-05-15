import rclpy
from rclpy.node import Node
import threading
import torch

from rosgraph_msgs.msg import Clock
from unitree_go.msg import LowCmd, LowState
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Twist
from go2_isaac_ros2.env import IsaacSimGo2EnvWrapper
from go2_isaac_ros2.lidar import get_head_lidar_pointcloud


class Go2SubNode(Node):
    def __init__(self, env: IsaacSimGo2EnvWrapper):
        super().__init__("go2_sub_node")

        self.lowcmd_sub = self.create_subscription(LowCmd, "/lowcmd", self.lowcmd_cb, 1)
        self.thead = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        self.env = env

    def lowcmd_cb(self, msg: LowCmd):
        action = torch.zeros(12)
        stiffness = torch.zeros(12)
        damping = torch.zeros(12)
        for i in range(12):
            action[i] = msg.motor_cmd[i].q
            stiffness[i] = msg.motor_cmd[i].kp
            damping[i] = msg.motor_cmd[i].kd
        self.env.set_action(action)
        self.env.set_stiffness(stiffness)
        self.env.set_damping(damping)

    def start(self):
        self.thead.start()


class Go2PubNode(Node):
    def __init__(self):
        super().__init__("go2_pub_node")

        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.low_state_pub = self.create_publisher(LowState, "/lowstate", 10)
        self.head_lidar_pub = self.create_publisher(PointCloud2, "utlidar/cloud", 10)
        self.ground_truth_twist_pub = self.create_publisher(
            Twist, "/ground_truth/twist", 10
        )
        self.clock_msg = None

    def publish(self, obs: dict, sim_time_sec: float):
        self._pub_clock(sim_time_sec)
        self._pub_low_state(obs)
        self._pub_head_lidar()
        self._pub_ground_truth(obs)

    def _pub_clock(self, sim_time_sec: float):
        msg = Clock()
        msg.clock = self.get_clock().now().to_msg()
        msg.clock.sec = int(sim_time_sec)
        msg.clock.nanosec = int((sim_time_sec - int(sim_time_sec)) * 1e9)
        self.clock_msg = msg
        self.clock_pub.publish(msg)

    def _pub_low_state(self, obs: dict):
        msg = LowState()

        for i in range(12):
            msg.motor_state[i].q = obs["obs"]["joint_pos"][0, i].item()
            msg.motor_state[i].dq = obs["obs"]["joint_vel"][0, i].item()

        msg.imu_state.quaternion[0] = obs["obs"]["imu_body_orientation"][0, 0].item()
        msg.imu_state.quaternion[1] = obs["obs"]["imu_body_orientation"][0, 1].item()
        msg.imu_state.quaternion[2] = obs["obs"]["imu_body_orientation"][0, 2].item()
        msg.imu_state.quaternion[3] = obs["obs"]["imu_body_orientation"][0, 3].item()
        msg.imu_state.gyroscope[0] = obs["obs"]["imu_body_ang_vel"][0, 0].item()
        msg.imu_state.gyroscope[1] = obs["obs"]["imu_body_ang_vel"][0, 1].item()
        msg.imu_state.gyroscope[2] = obs["obs"]["imu_body_ang_vel"][0, 2].item()
        msg.imu_state.accelerometer[0] = obs["obs"]["imu_body_lin_acc"][0, 0].item()
        msg.imu_state.accelerometer[1] = obs["obs"]["imu_body_lin_acc"][0, 1].item()
        msg.imu_state.accelerometer[2] = obs["obs"]["imu_body_lin_acc"][0, 2].item()

        self.low_state_pub.publish(msg)

    def _pub_head_lidar(self):
        pcl = get_head_lidar_pointcloud().tolist()

        header = Header()
        header.frame_id = "utlidar_lidar"
        header.stamp.sec = self.clock_msg.clock.sec
        header.stamp.nanosec = self.clock_msg.clock.nanosec

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name="intensity", offset=16, datatype=PointField.FLOAT32, count=1
            ),
            PointField(name="ring", offset=20, datatype=PointField.UINT16, count=1),
            PointField(name="time", offset=24, datatype=PointField.FLOAT32, count=1),
        ]

        pcl_msg = point_cloud2.create_cloud(header, fields, pcl)
        self.head_lidar_pub.publish(pcl_msg)

    def _pub_ground_truth(self, obs: dict):
        msg = Twist()
        msg.linear.x = obs["obs"]["body_lin_vel"][0, 0].item()
        msg.linear.y = obs["obs"]["body_lin_vel"][0, 1].item()
        msg.linear.z = obs["obs"]["body_lin_vel"][0, 2].item()
        msg.angular.x = obs["obs"]["body_ang_vel"][0, 0].item()
        msg.angular.y = obs["obs"]["body_ang_vel"][0, 1].item()
        msg.angular.z = obs["obs"]["body_ang_vel"][0, 2].item()
        self.ground_truth_twist_pub.publish(msg)

    def _clock_to_sec(self, clock_msg: Clock) -> float:
        return clock_msg.clock.sec + clock_msg.clock.nanosec / 1e9
