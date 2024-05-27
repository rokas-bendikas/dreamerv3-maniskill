from collections import OrderedDict

import gymnasium
import numpy as np
from envs.exceptions import UnknownTaskError
from gymnasium.spaces import Box, Dict
from mani_skill.utils.wrappers.gymnasium import ManiSkillCPUGymWrapper

# import mani_skill.envs
from mani_skill.utils.common import (
    flatten_dict_keys,
)

import sapien
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


MANISKILL_TASKS = [
    "PushCube-v1-multicam",
    "StackCube-v1-multicam",
    "PickCube-v1-multicam",
    "PushCube-v1-multicam-groundless",
    "StackCube-v1-multicam-groundless",
    "PickCube-v1-multicam-groundless",
]

from mani_skill.envs.tasks import (
    PushCubeEnv,
    PickCubeEnv as PickCubeEnvOriginal,
    StackCubeEnv,
)


class PickCubeEnv(PickCubeEnvOriginal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self._hidden_objects.remove(self.goal_site)


SUPPORTED_ENVS = [
    PushCubeEnv,
    StackCubeEnv,
    PickCubeEnv,
]


class MulticamTask:

    def __init__(
        self,
        randomize_cameras,
        num_additional_cams,
        cam_resolution,
        near_far,
        scene_center=[0, 0, 0.1],
    ):
        self._randomize_cameras = randomize_cameras
        self._num_additional_cams = num_additional_cams
        self._camera_resolutions = cam_resolution
        self._near_far = near_far
        self.cam_names = ["cam_wrist"] + [
            f"cam_additional_{i}" for i in range(num_additional_cams)
        ]
        self._center = scene_center

    def _sample_additional_camera_position(self):
        """Samples a random pose of a camera on the upper hemisphere."""

        radius_limits = [0.4, 0.5]
        radius = np.random.uniform(*radius_limits)

        # Adjust the camera position horizontally.
        phi = np.random.uniform(-0.3 * np.pi, 0.3 * np.pi)
        # Adjust the camera elevation.
        cos_theta = np.random.uniform(0.0, 1.0)
        theta = np.arccos(cos_theta)

        # Spherical to Cartesian conversion.
        x = radius * np.sin(theta) * np.cos(phi) + self._center[0]
        y = radius * np.sin(theta) * np.sin(phi) + self._center[1]
        z = radius * np.cos(theta) + self._center[2]
        pos = [x, y, z]
        return pos

    @property
    def _default_sensor_configs(self):
        # Calculate the intrinsic matrix
        FOV = np.pi / 2
        cam_list = [
            CameraConfig(
                "cam_wrist",
                Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                height=self._camera_resolutions[0],
                width=self._camera_resolutions[1],
                intrinsic=None,
                fov=FOV,
                near=0.01,
                far=100,
                mount=sapien_utils.get_obj_by_name(
                    self.agent.robot.links, "camera_link"
                ),
            ),
            #     CameraConfig(
            #         "base_camera",
            #         sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1]),
            #         128,
            #         128,
            #         np.pi / 2,
            #         0.01,
            #         100,
            #     ),
        ]

        for i in range(self._num_additional_cams):
            cam_pose = sapien_utils.look_at(
                eye=self._sample_additional_camera_position(), target=[0, 0, 0.1]
            )
            cam_list.append(
                CameraConfig(
                    f"cam_additional_{i}",
                    cam_pose,
                    height=self._camera_resolutions[0],
                    width=self._camera_resolutions[1],
                    intrinsic=None,
                    fov=FOV,
                    near=self._near_far[0],
                    far=self._near_far[1],
                    mount=None,
                )
            )
        return cam_list


def build_fake_ground(scene, floor_width=20, altitude=0, name="ground"):
    ground = scene.create_actor_builder()
    ground.add_plane_collision(
        pose=sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0, -0.7071068, 0]),
    )
    ground.add_plane_visual(
        pose=sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0, -0.7071068, 0]),
        scale=(floor_width, floor_width, floor_width),
        material=sapien.render.RenderMaterial(
            base_color=[0.9, 0.9, 0.93, 0], metallic=0.5, roughness=0.5
        ),
    )
    return ground.build_static(name=name)


def register_multicam_env(env_name, max_episode_steps):
    def decorator(some_base_env):
        class MulticamEnv(MulticamTask, some_base_env):
            SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch", "panda_wristcam"]

            def __init__(self, *args, **kwargs):
                MulticamTask.__init__(
                    self,
                    kwargs.pop("randomize_cameras"),
                    kwargs.pop("num_additional_cams"),
                    kwargs.pop("cam_resolution"),
                    kwargs.pop("near_far"),
                )
                some_base_env.__init__(
                    self,
                    robot_uids="panda_wristcam",
                    *args,
                    reconfiguration_freq=1 if self._randomize_cameras else 0,
                    **kwargs,
                )

        register_env(env_name, max_episode_steps)(MulticamEnv)
        return MulticamEnv

    return decorator


def register_multicam_groundless_env(env_name, max_episode_steps):
    def decorator(some_base_env):
        class MulticamGroundlessEnv(MulticamTask, some_base_env):
            SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch", "panda_wristcam"]

            def __init__(self, *args, **kwargs):
                MulticamTask.__init__(
                    self,
                    kwargs.pop("randomize_cameras"),
                    kwargs.pop("num_additional_cams"),
                    kwargs.pop("cam_resolution"),
                    kwargs.pop("near_far"),
                )
                some_base_env.__init__(
                    self,
                    robot_uids="panda_wristcam",
                    *args,
                    reconfiguration_freq=1 if self._randomize_cameras else 0,
                    **kwargs,
                )

            def _load_scene(self, options: dict):
                super()._load_scene(options)
                self.table_scene.ground.remove_from_scene()
                self.table_scene.ground = build_fake_ground(
                    self.table_scene.scene, name="fake-ground"
                )

        register_env(env_name, max_episode_steps)(MulticamGroundlessEnv)
        return MulticamGroundlessEnv

    return decorator


for env in SUPPORTED_ENVS:
    env_name = env.__name__.split("Env")[0]

    register_multicam_env(f"{env_name}-v1-multicam", max_episode_steps=50)(env)
    register_multicam_groundless_env(
        f"{env_name}-v1-multicam-groundless", max_episode_steps=50
    )(env)


class ManiSkillObsWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Dict()
        obs, _ = self.env.reset()
        sensor_data = flatten_dict_keys(obs["sensor_data"])
        obs_wrist = Box(
            0, 255, shape=sensor_data["cam_wrist/rgb"].shape, dtype=np.uint8
        )
        obs_cam1 = Box(
            0, 255, shape=sensor_data["cam_additional_0/rgb"].shape, dtype=np.uint8
        )
        obs_cam2 = Box(
            0, 255, shape=sensor_data["cam_additional_1/rgb"].shape, dtype=np.uint8
        )
        self.observation_space.spaces["wrist_rgb"] = obs_wrist
        self.observation_space.spaces["cam1_rgb"] = obs_cam1
        self.observation_space.spaces["cam2_rgb"] = obs_cam2

    def observation(self, observation):
        sensor_data = flatten_dict_keys(observation["sensor_data"])
        return OrderedDict(
            wrist_rgb=sensor_data["cam_wrist/rgb"],
            cam1_rgb=sensor_data["cam_additional_0/rgb"],
            cam2_rgb=sensor_data["cam_additional_1/rgb"],
        )


class ManiSkillGymnasium2GymWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return obs

    def step(self, action):
        obs, reward, terminate, truncate, info = self.env.step(action)
        done = terminate or truncate
        info.update({"is_terminal": terminate})
        return obs, reward, done, info


def make_maniskill_env(cfg):
    """
    Make ManiSkill3 environment.
    """
    if cfg.task not in MANISKILL_TASKS:
        raise UnknownTaskError(cfg.task)
    env = gymnasium.make(
        cfg.task,
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        randomize_cameras=True,
        num_additional_cams=2,
        near_far=[0.00001, 2.0],
        cam_resolution=[64, 64],
    )
    env = ManiSkillCPUGymWrapper(env)
    env = ManiSkillObsWrapper(env)
    env = ManiSkillGymnasium2GymWrapper(env)
    return env
