#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team.

import sys
from enum import IntEnum
from typing import Any

import numpy as np

from ..teleoperator import Teleoperator
from .configuration_phone import PhoneTeleopConfig

# --------------------------------------------------------------------- #
#  ENUM / MAP FOR GRIPPER
# --------------------------------------------------------------------- #
class GripperAction(IntEnum):
    CLOSE = 0
    STAY  = 1
    OPEN  = 2


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open":  GripperAction.OPEN.value,
    "stay":  GripperAction.STAY.value,
}

# --------------------------------------------------------------------- #
#  BASE TELEOP — Δx Δy Δz (+ gripper)                                  #
# --------------------------------------------------------------------- #
class PhoneTeleop(Teleoperator):
    """
    Teleoperator that converts phone‑generated IMU deltas into
    Cartesian position commands (Δx, Δy, Δz) plus an optional gripper
    signal.  Uses IMUController (HTTP/UDP) under the hood.
    """

    config_class = PhoneTeleopConfig
    name = "phone"

    # -------- life‑cycle ------------------------------------------------
    def __init__(self, config: PhoneTeleopConfig):
        super().__init__(config)
        self.config      = config
        self.robot_type  = config.type
        self.gamepad     = None       # set in connect()


    def connect(self) -> None:
        """
        Create the low‑level driver (IMUController) and start it.
        For historical reasons we still keep the macOS HID path, but
        most users will go through the HTTP IMU driver.
        """
        if sys.platform == "darwin":
            from .phone_utils import IMUController as Driver
        else:
            from .phone_utils import IMUController as Driver   # same driver on non‑mac

        self.gamepad = Driver(
            x_step_size=self.config.x_step_size,
            y_step_size=self.config.y_step_size,
            z_step_size=self.config.z_step_size,
            port=self.config.port,
        )
        self.gamepad.start()

    def disconnect(self) -> None:
        if self.gamepad:
            self.gamepad.stop()
            self.gamepad = None

    # -------- schema descriptors ---------------------------------------
    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype":  "float32",
                "shape":  (4,),
                "names":  {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        return {
            "dtype": "float32",
            "shape": (3,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    # -------- polling ---------------------------------------------------
    def get_action(self) -> dict[str, Any]:
        if not self.gamepad:
            raise RuntimeError("PhoneTeleop not connected")

        self.gamepad.update()
        dx, dy, dz = self.gamepad.get_deltas()

        # print("**************[get_action] dx, dy and dz is:", dx, dy, dz)
        action_dict: dict[str, Any] = {
            "delta_x": float(dx),
            "delta_y": float(dy),
            "delta_z": float(dz),

        }

        print("Action dict pulled is:", action_dict)

        if self.config.use_gripper:
            cmd  = self.gamepad.gripper_command()            # "open" | "close" | "stay"
            gval = gripper_action_map[cmd]
            action_dict["gripper"] = gval

        return action_dict

    # -------- misc / passthrough ---------------------------------------
    def is_connected(self) -> bool:
        return self.gamepad is not None

    def calibrate(self)        -> None: pass
    def is_calibrated(self)    -> bool: return True
    def configure(self)        -> None: pass
    def send_feedback(self, *_): pass

# --------------------------------------------------------------------- #
#  EXTENDED TELEOP — Δx Δy Δz Δroll Δpitch Δyaw (+ gripper)             #
# --------------------------------------------------------------------- #
class PhoneEndEffectorTeleop(PhoneTeleop):
    """
    Same as PhoneTeleop but also outputs orientation deltas
    (roll, pitch, yaw) when the IMU driver provides them.
    Expected extra driver method:  get_rot_deltas() -> (dr, dp, dy)
    """

    name = "phone_eef"

    # ------ schema with 6‑DoF (+ gripper) ------------------------------
    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (7,),
                "names": {
                    "delta_x": 0, "delta_y": 1, "delta_z": 2,
                    "delta_roll": 3, "delta_pitch": 4, "delta_yaw": 5,
                    "gripper": 6,
                },
            }
        return {
            "dtype": "float32",
            "shape": (6,),
            "names": {
                "delta_x": 0, "delta_y": 1, "delta_z": 2,
                "delta_roll": 3, "delta_pitch": 4, "delta_yaw": 5,
            },
        }

    # ------ action polling ---------------------------------------------
    def get_action(self) -> dict[str, Any]:
        if not self.gamepad:
            raise RuntimeError("PhoneEndEffectorTeleop not connected")

        self.gamepad.update()
        dx, dy, dz = self.gamepad.get_deltas()

        # Orientation deltas: fall back to zeros if driver lacks method
        if hasattr(self.gamepad, "get_rot_deltas"):
            dr, dp, dyaw = self.gamepad.get_rot_deltas()
        else:
            dr = dp = dyaw = 0.0

        action = {
            "delta_x":    float(dx),
            "delta_y":    float(dy),
            "delta_z":    float(dz),
            "delta_roll": float(dr),
            "delta_pitch":float(dp),
            "delta_yaw":  float(dyaw),
        }

        if self.config.use_gripper:
            cmd  = self.gamepad.gripper_command()
            gval = gripper_action_map[cmd]
            action["gripper"] = gval

        return action
