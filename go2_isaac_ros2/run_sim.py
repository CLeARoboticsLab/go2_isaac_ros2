# start Isaac Sim
print("Starting Isaac Sim")
from isaaclab.app import AppLauncher

app_launcher = AppLauncher()
simulation_app = app_launcher.app

import omni
from go2_isaac_ros2.env import UnitreeGo2CustomEnvCfg, IsaacSimGo2EnvWrapper
from isaaclab.envs import ManagerBasedEnv
from go2_isaac_ros2.lidar import add_head_lidar
import rclpy
from go2_isaac_ros2.ros import Go2PubNode, Go2SubNode
import time


def run_sim():
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
    timeline = omni.timeline.get_timeline_interface()

    # create environment
    env_cfg = UnitreeGo2CustomEnvCfg()
    env = ManagerBasedEnv(env_cfg)
    env = IsaacSimGo2EnvWrapper(env)
    add_head_lidar()

    # reset environment
    obs, _ = env.reset()

    # start ros2 nodes
    rclpy.init()
    go2_pub_node = Go2PubNode()
    go2_sub_node = Go2SubNode(env)
    go2_sub_node.start()

    while simulation_app.is_running():
        start_time = time.time()
        obs, _ = env.step()
        sim_time_sec = timeline.get_current_time()
        go2_pub_node.publish(obs, sim_time_sec)

        # time delay to keep simulation at or below real time
        sleep_time = env.dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    rclpy.shutdown()
