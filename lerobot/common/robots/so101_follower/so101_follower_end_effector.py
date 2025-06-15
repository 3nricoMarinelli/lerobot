# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing peselfrmissions and
# limitations under the License.

import logging
import time
from typing import Any
from math import atan2, degrees, hypot

import numpy as np

from lerobot.common.cameras import make_cameras_from_configs
from lerobot.common.errors import DeviceNotConnectedError
from lerobot.common.model.kinematics import RobotKinematics
from lerobot.common.motors import Motor, MotorNormMode
from lerobot.common.motors.feetech import FeetechMotorsBus

from . import SO101Follower
from .config_so101_follower import SO101FollowerEndEffectorConfig

logger = logging.getLogger(__name__)
EE_FRAME = "gripper_tip"


class SO101FollowerEndEffector(SO101Follower):
    """
    SO101Follower robot with end-effector space control.

    This robot inherits from SO100Follower but transforms actions from
    end-effector space to joint space before sending them to the motors.
    """

    config_class = SO101FollowerEndEffectorConfig
    name = "so101_follower_end_effector"

    def __init__(self, config: SO101FollowerEndEffectorConfig):
        super().__init__(config)
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        self.cameras = make_cameras_from_configs(config.cameras)

        self.config = config

        # Initialize the kinematics module for the so100 robot
        self.kinematics = RobotKinematics(robot_type="so_new_calibration")

        # Store the bounds for end-effector position
        self.end_effector_bounds = self.config.end_effector_bounds

        self.ee_pose = None
        self.joint_pos = None

    @property
    def action_features(self) -> dict[str, Any]:
        """
        Define action features for end-effector control.
        Returns dictionary with dtype, shape, and names.
        """
        return {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "x": 0,
            "y": 1,
            "z": 2,
            "roll": 3,
            "pitch": 4,
            "yaw": 5,
            "gripper": 6
            }
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Transform action from end-effector space to joint space and send to motors.

        Args:
            action: Dictionary with keys 'x', 'y', 'z' for end-effector control
                   or a numpy array with [x, y, z]

        Returns:
            The joint-space action that was sent to the motors
        """

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        print(f"++++++++++++[DEBUG ] Action received",action)

        # Convert action to numpy array if not already
        if isinstance(action, dict):
            if all(k in action for k in ["x", "y", "z", "roll", "pitch", "yaw"]) and self.ee_pose is not None:
                if "gripper" not in action:
                    action["gripper"] = 1.0
                    logger.warning(
                    f"No gripper action provided, defaulting to 1.0 (open).")
                
                action = np.array([
                    (self.ee_pose[0, 3] - action["x"]) * self.config.end_effector_step_sizes["x"],
                    (self.ee_pose[1, 3] - action["y"]) * self.config.end_effector_step_sizes["y"],
                    (self.ee_pose[2, 3] - action["z"]) * self.config.end_effector_step_sizes["z"],
                    (degrees(atan2(self.ee_pose[2,1], self.ee_pose[2,2])) - action["roll"]) * self.config.end_effector_step_sizes["roll"],
                    (degrees(-self.ee_pose[2,0], hypot(self.ee_pose[2,1], self.ee_pose[2,2])) - action["roll"]) * self.config.end_effector_step_sizes["pitch"],
                    (degrees(self.ee_pose[1,0], self.ee_pose[0,0]) - action["roll"]) * self.config.end_effector_step_sizes["yaw"],
                    action["gripper"]
                ], dtype=np.float32)
            else:
                if not all(k in action for k in ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]):
                    logger.warning(
                        f"Expected action keys 'x', 'y', 'z', "
                        f"'roll', 'pitch', 'yaw' got {list(action.keys())}"
                    )
                action = np.zeros(7, dtype=np.float32)

        print(f"++++++++++++[DEBUG ] Actual action sent",action)

        if self.joint_pos is None:
            # Read current joint positions
            joint_pos = self.bus.sync_read("Present_Position")
            self.joint_pos = np.array([joint_pos[name] for name in self.bus.motors])

        # Calculate current end-effector position using forward kinematics
        if self.ee_pose is None:
            self.ee_pose = self.kinematics.forward_kinematics(self.joint_pos, frame=EE_FRAME)

        # Set desired end-effector position by adding delta
        desired_ee_pos = np.eye(4)

        # Convert rotation deltas from degrees to radians
        roll = np.radians(action[3])
        pitch = np.radians(action[4])
        yaw = np.radians(action[5])

        # Create rotation matrices for each axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Combine rotations (applying them in order: roll, pitch, yaw)
        desired_ee_pos[:3, :3] = Rz @ Ry @ Rx

        # delta_rotation = current_rot @ desired_ee_pos

        # Add delta to position and clip to bounds
        desired_ee_pos[:3, 3] = action[:3]
        if self.end_effector_bounds is not None:
            desired_ee_pos[:3, 3] = np.clip(
                desired_ee_pos[:3, 3],
                self.end_effector_bounds["min"],
                self.end_effector_bounds["max"],
            )

        # Compute inverse kinematics to get joint positions
        target_joint_values_in_degrees = self.kinematics.ik(
            self.joint_pos, desired_ee_pos, position_only=True, frame=EE_FRAME
        )

        target_joint_values_in_degrees = np.clip(target_joint_values_in_degrees, -180.0, 180.0)
        # Create joint space action dictionary
        joint_action = {
            f"{key}.pos": target_joint_values_in_degrees[i] for i, key in enumerate(self.bus.motors.keys())
        }

        # Handle gripper separately if included in action
        # Gripper delta action is in the range 0 - 2,
        # We need to shift the action to the range -1, 1 so that we can expand it to -Max_gripper_pos, Max_gripper_pos
        joint_action["gripper.pos"] = np.clip(
            self.joint_pos[-1] + (action[-1] - 1) * self.config.max_gripper_pos,
            5,
            self.config.max_gripper_pos,
        )

        self.ee_pose = desired_ee_pos.copy()
        self.joint_pos = target_joint_values_in_degrees.copy()
        self.joint_pos[-1] = joint_action["gripper.pos"]

        # DEBUG Prints: 
        

        print(f"[DEBUG send_action] goal_pos to write: {joint_action}")

        # Send joint space action to parent class
        return super().send_action(joint_action)
    


    

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read aselfrm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def reset(self):
        self.ee_pose = None
        self.joint_pos = None
