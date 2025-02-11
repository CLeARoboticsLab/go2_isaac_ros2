# start Isaac Sim
print("Starting Isaac Sim")
from isaaclab.app import AppLauncher

app_launcher = AppLauncher()
simulation_app = app_launcher.app

import omni
from go2_isaac_ros2.env import UnitreeGo2CustomEnvCfg


def run_sim():
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

    env_cfg = UnitreeGo2CustomEnvCfg()

    while simulation_app.is_running():
        pass
