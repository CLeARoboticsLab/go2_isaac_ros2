from dataclasses import MISSING
from typing import Literal
import torch
import omni.kit.commands
import omni.replicator.core as rep
from pxr import Gf

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ImuCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]
STANDING_JOINT_ANGLES = [0.0, 0.67, -1.3] * 4
PRONE_JOINT_ANGLES = [
    -0.35,
    1.36,
    -2.65,
    0.35,
    1.36,
    -2.65,
    -0.5,
    1.36,
    -2.65,
    0.5,
    1.36,
    -2.65,
]
JOINT_STIFFNESS = 100.0

action_global = torch.tensor(PRONE_JOINT_ANGLES, dtype=torch.float32).unsqueeze(0)


def set_action(action: torch.Tensor) -> None:
    global action_global
    action_global = action


def get_action(env: ManagerBasedEnvCfg) -> torch.Tensor:
    global action_global
    return action_global.to(env.device)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        debug_vis=False,
    )

    # robots
    robot: ArticulationCfg = MISSING

    # sensors
    imu_body = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=ImuCfg.OffsetCfg(pos=(-0.02557, 0.0, 0.04232)),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

    # add a block for now so that LIO works
    block = AssetBaseCfg(
        prim_path="/World/block",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.5, 0.75),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.0, 0.0), metallic=0.2
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 2.0, 1.0)),
    )


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    eye: tuple[float, float, float] = (7.5, 7.5, 7.5)
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cam_prim_path: str = "/OmniverseKit_Persp"
    resolution: tuple[int, int] = (1920, 1080)
    origin_type: Literal["world", "env", "asset_root"] = "world"
    env_index: int = 0
    asset_name: str | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ObsCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot", joint_names=JOINT_NAMES, preserve_order=True
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot", joint_names=JOINT_NAMES, preserve_order=True
                )
            },
        )
        imu_body_orientation = ObsTerm(
            func=mdp.imu_orientation,
            params={
                "asset_cfg": SceneEntityCfg(name="imu_body"),
            },
        )
        imu_body_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={
                "asset_cfg": SceneEntityCfg(name="imu_body"),
            },
        )
        imu_body_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc,
            params={
                "asset_cfg": SceneEntityCfg(name="imu_body"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    obs: ObsCfg = ObsCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        preserve_order=True,
        use_default_offset=False,
    )


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=2.5)
    viewer: ViewerCfg = ViewerCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        # simulation settings
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material


@configclass
class UnitreeGo2CustomEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        dc_motor_cfg = UNITREE_GO2_CFG.actuators["base_legs"]
        dc_motor_cfg = dc_motor_cfg.replace(
            joint_names_expr=JOINT_NAMES,
            stiffness=JOINT_STIFFNESS,
        )

        self.scene.robot = UNITREE_GO2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators={"base_legs": dc_motor_cfg},
        )


def add_head_lidar():
    _, head_lidar = omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar",
        path="head_lidar",
        parent="/World/envs/env_0/Robot/base",
        translation=(0.36, 0.0, -0.12),  # is outside of the base mesh
        orientation=Gf.Quatd(0.1313147, 0, 0.9913407, 0),
        config="Unitree_L1",
    )

    render_product = rep.create.render_product(head_lidar.GetPath(), [1, 1])

    annotator = rep.AnnotatorRegistry.get_annotator(
        "RtxSensorCpuIsaacCreateRTXLidarScanBuffer"
    )
    annotator.attach(render_product)

    return annotator
