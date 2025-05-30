from dataclasses import MISSING
from typing import Literal
import torch
import threading

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnvCfg, ManagerBasedEnv, VecEnvObs
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ImuCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


SIM_DT = 0.005
SIM_RENDER_INTERVAL = 4
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
JOINT_STIFFNESS = 75.0
JOINT_DAMPING = 0.5


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
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

    # add a blocks for now so that LIO works
    block1 = AssetBaseCfg(
        prim_path="/World/block1",
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
    block2 = AssetBaseCfg(
        prim_path="/World/block2",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 2.5, 0.75),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.8), metallic=0.2
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.5, -4.5, 1.0)),
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

        body_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        body_ang_vel = ObsTerm(func=mdp.base_ang_vel)

        world_pos = ObsTerm(func=mdp.root_pos_w)
        world_quat = ObsTerm(func=mdp.root_quat_w)
        world_lin_vel = ObsTerm(func=mdp.root_lin_vel_w)
        world_ang_vel = ObsTerm(func=mdp.root_ang_vel_w)

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
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),  # Friction will be deterministic
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=2.5)
    viewer: ViewerCfg = ViewerCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        # simulation settings
        self.sim.dt = SIM_DT
        self.sim.render_interval = SIM_RENDER_INTERVAL
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
            damping=JOINT_DAMPING,
        )

        self.scene.robot = UNITREE_GO2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators={"base_legs": dc_motor_cfg},
        )


class IsaacSimGo2EnvWrapper:
    def __init__(self, env: ManagerBasedEnv):
        self._env = env
        self.device = self._env.device
        self.lock = threading.Lock()

        self.action = torch.tensor(PRONE_JOINT_ANGLES, dtype=torch.float32)
        self.set_action(self.action)

        self.joint_ids = torch.zeros(12, dtype=torch.int32)
        for i, name in enumerate(JOINT_NAMES):
            self.joint_ids[i] = env.scene.articulations["robot"].find_joints(name)[0][0]
        self.set_stiffness(torch.tensor([JOINT_STIFFNESS] * 12, dtype=torch.float32))
        self.set_damping(torch.tensor([JOINT_DAMPING] * 12, dtype=torch.float32))

    def reset(self):
        return self._env.reset()

    def step(self, action_unused=None) -> tuple[VecEnvObs, dict]:
        with self.lock:
            with torch.inference_mode():
                return self._env.step(self.action)

    def close(self):
        return self._env.close()

    def set_action(self, action: torch.Tensor):
        with self.lock:
            self.action = action.unsqueeze(0).to(self.device)

    def set_stiffness(self, stiffness: torch.Tensor):
        with self.lock:
            stiffness = stiffness[self.joint_ids]
            stiffness = stiffness.unsqueeze(0).to(self.device)
            self._env.scene.articulations["robot"].actuators[
                "base_legs"
            ].stiffness = stiffness

    def set_damping(self, damping: torch.Tensor):
        with self.lock:
            damping = damping[self.joint_ids]
            damping = damping.unsqueeze(0).to(self.device)
            self._env.scene.articulations["robot"].actuators[
                "base_legs"
            ].damping = damping

    @property
    def dt(self) -> float:
        return self._env.physics_dt
