# start Isaac Sim
print("Starting Isaac Sim")
from isaaclab.app import AppLauncher

app_launcher = AppLauncher()
simulation_app = app_launcher.app

import omni
import go2_isaac_ros2.env
from go2_isaac_ros2.env import UnitreeGo2CustomEnvCfg
from isaaclab.envs import ManagerBasedEnv
from go2_isaac_ros2.lidar import add_head_lidar
import rclpy
from go2_isaac_ros2.ros import add_lowcmd_sub, Go2PubNode


def run_sim():
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
    timeline = omni.timeline.get_timeline_interface()

    # create environment
    env_cfg = UnitreeGo2CustomEnvCfg()
    env = ManagerBasedEnv(env_cfg)
    add_head_lidar()

    # reset environment
    obs, _ = env.reset()

    # start ros2 nodes
    rclpy.init()
    add_lowcmd_sub()
    go2_pub_node = Go2PubNode()

    while simulation_app.is_running():
        action = go2_isaac_ros2.env.get_action(env)
        obs, _ = env.step(action)
        sim_time_sec = timeline.get_current_time()
        go2_pub_node.publish(obs, sim_time_sec)
