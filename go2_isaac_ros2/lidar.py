import numpy as np
import omni.kit.commands
from pxr import UsdGeom
from isaacsim.sensors.physx import _range_sensor


LIDAR_PARENT = "/World/envs/env_0/Robot/base/radar"
LIDAR_PRIM = "head_lidar"
LIDAR_PATH = f"{LIDAR_PARENT}/{LIDAR_PRIM}"
LIDAR_SCAN_FREQ = 180.0
LIDAR_H_RES = 2.0
LIDAR_V_RES = 2.0
LIDAR_Z_OFFSET = 0.06

lidarInterface = _range_sensor.acquire_lidar_sensor_interface()


def add_head_lidar():
    _, prim = omni.kit.commands.execute(
        "RangeSensorCreateLidar",
        path=LIDAR_PRIM,
        parent=LIDAR_PARENT,
        min_range=0.05,
        max_range=30.0,
        draw_points=True,
        draw_lines=False,
        horizontal_fov=360.0,
        vertical_fov=180.0,
        horizontal_resolution=LIDAR_H_RES,
        vertical_resolution=LIDAR_V_RES,
        rotation_rate=LIDAR_SCAN_FREQ,
        high_lod=True,
        yaw_offset=0.0,
        enable_semantics=False,
    )

    # translate the lidar from radar frame to be outside the base mesh
    UsdGeom.XformCommonAPI(prim).SetTranslate((0.0, 0.0, LIDAR_Z_OFFSET))


def get_head_lidar_pointcloud():
    points = lidarInterface.get_point_cloud_data(LIDAR_PATH)
    points[:, :, 2] += LIDAR_Z_OFFSET  # translate to the radar frame
    depths = lidarInterface.get_linear_depth_data(LIDAR_PATH)

    H = points.shape[0]
    V = points.shape[1]

    # assume each vertical scan is a unique ID
    beam_ids = np.tile(np.arange(V), (H, 1))

    # each vertical scan is a separate timestamp; assume timestamps are
    # evenly spaced across a timestep
    sweep_deg = (H - 1) * LIDAR_H_RES
    t = sweep_deg / 360.0 / LIDAR_SCAN_FREQ
    timestamps = np.tile(
        np.linspace(0, t, H, endpoint=False)[:, np.newaxis],
        (1, V),
    )

    # flatten to (H*V, dim)
    points = points.reshape(-1, 3)
    depths = depths.reshape(-1)
    beam_ids = beam_ids.reshape(-1)
    timestamps = timestamps.reshape(-1)

    # remove points with no hits
    intens_mask = lidarInterface.get_intensity_data(LIDAR_PATH).reshape(-1)
    intens_mask = (intens_mask > 0).astype(np.bool_)

    # remove points with zenith angle outside of [-pi/2, 0], since we used
    # a 180-degree vertical FOV
    zenith = lidarInterface.get_zenith_data(LIDAR_PATH)[np.newaxis, :]  # (1, V)
    zenith_mask = np.tile((zenith >= -np.pi / 2) & (zenith <= 0), (H, 1))  # (H, V)
    zenith_mask = zenith_mask.reshape(-1)  # Flatten to (H*V,)

    mask = intens_mask & zenith_mask
    points = points[mask]
    depths = depths[mask]
    beam_ids = beam_ids[mask]
    timestamps = timestamps[mask]

    # assume intensity is inversely proportional to the square of depth
    intensities = 100 * 1 / (1 + depths**2)

    pcl = np.column_stack((points, intensities, beam_ids, timestamps))

    return pcl
