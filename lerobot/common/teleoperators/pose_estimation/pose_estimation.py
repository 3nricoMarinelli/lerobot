#!/usr/bin/env python

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
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any
import socket
import struct
import numpy as np
import multiprocessing
import sys
import os
from pathlib import Path

from lerobot.common.errors import DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_pose_estimation import PoseEstimationConfig

logger = logging.getLogger(__name__)

def run_detect_hand():
    """Function to run detect_hand script in a separate process"""
    current_dir = Path(__file__).parent
    detect_hand_path = current_dir / "detect_hand.py"
    os.system(f"python {detect_hand_path}")

class PoseEstimation(Teleoperator):
    """
    Pose Estimation Arm designed by TheRobotStudio and Hugging Face.
    """

    config_class = PoseEstimationConfig

    def __init__(self, config: PoseEstimationConfig):
        super().__init__(config)
        self.config = config
        self.hand_detection_process = None
        self.sock = None

    def connect(self) -> bool:
        """Connect to the device and start hand detection"""
        try:
            # Setup socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.receiver_address = ('localhost', 11310)
            self.sock.bind(self.receiver_address)
            
            # Start hand detection process
            self.hand_detection_process = multiprocessing.Process(
                target=run_detect_hand
            )
            self.hand_detection_process.start()
            logger.info("Hand detection started successfully")
            
            return super().connect()
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from the device and stop hand detection"""
        try:
            if self.hand_detection_process and self.hand_detection_process.is_alive():
                self.hand_detection_process.terminate()
                self.hand_detection_process.join()
                logger.info("Hand detection stopped successfully")
            
            if self.sock:
                self.sock.close()
                self.sock = None
            
            return super().disconnect()
        except Exception as e:
            logger.error(f"Failed to disconnect: {e}")
            return False

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "x.pose": 0,
                "y.pose": 1,
                "z.pose": 2,
                "roll.pose": 3,
                "pitch.pose": 4,
                "yaw.pose": 5,
                # "delta_gripper": 6
            },
        }

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "Pose Estimation is not connected. You need to run `connect()` before `get_action()`."
            )

        # call the pose estimation model to get the action
        try:
            data, _ = self.sock.recvfrom(1024)
            # Unpack 15 floats: 9 for rotation matrix, 3 for position, 3 for euler angles
            unpacked = struct.unpack('15f', data)
            
            # Reconstruct data
            position = np.array(unpacked[:3])
            euler_angles = np.array(unpacked[3:6])
        except socket.error as e:
            logger.error(f"Socket error: {e}")
            raise DeviceNotConnectedError("Failed to receive data from the pose estimation model.")

        action_dict = {
            "x.pose": position[0],
            "y.pose": position[1],
            "z.pose": position[2],
            "roll.pose": euler_angles[0],
            "pitch.pose": euler_angles[1],
            "yaw.pose": euler_angles[2]
        }

        return action_dict
