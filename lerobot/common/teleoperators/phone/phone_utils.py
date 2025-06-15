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
import threading                        
from collections import deque            
from flask import Flask, request         


class InputController:
    """Base class for input controllers that generate motion deltas."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        """
        Initialize the controller.

        Args:
            x_step_size: Base movement step size in meters
            y_step_size: Base movement step size in meters
            z_step_size: Base movement step size in meters
        """
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True
        self.episode_end_status = None  # None, "success", or "failure"
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False
        self.gyro_scale = 1.0      # tune this as needed
        self.acc_gain = 1.0        # tune this as needed
        self.gravity_z = 9.81      # typical gravity in m/s^2
        self.dt = 0.1              # ~time between sensor readings, e.g. 0.1 sec

    def start(self):
        """Start the controller and initialize resources."""
        pass

    def stop(self):
        """Stop the controller and release resources."""
        pass

    def get_deltas(self):
        """Get the current movement deltas (dx, dy, dz) in meters."""
        return 0.0, 0.0, 0.0

    def should_quit(self):
        """Return True if the user has requested to quit."""
        return not self.running

    def update(self):
        """Update controller state - call this once per frame."""
        pass

    def __enter__(self):
        """Support for use in 'with' statements."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are released when exiting 'with' block."""
        self.stop()

    def get_episode_end_status(self):
        """
        Get the current episode end status.

        Returns:
            None if episode should continue, "success" or "failure" otherwise
        """
        status = self.episode_end_status
        self.episode_end_status = None  # Reset after reading
        return status

    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.intervention_flag

    def gripper_command(self):
        """Return the current gripper command."""
        if self.open_gripper_command == self.close_gripper_command:
            return "stay"
        elif self.open_gripper_command:
            return "open"
        elif self.close_gripper_command:
            return "close"

class IMUController(InputController):
    # Check if the porte init is not interferring with the configuration_phone.py
    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01, port=5010):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.port = port

        # Buffers for incoming IMU data
        self.N = 5
        self.imu_data = {
            "ax": deque(maxlen=self.N),
            "ay": deque(maxlen=self.N),
            "az": deque(maxlen=self.N),
            "gx": deque(maxlen=self.N),
            "gy": deque(maxlen=self.N),
            "gz": deque(maxlen=self.N),

        }

        # Flask app for receiving data
        self.app = Flask(__name__)
        self._setup_routes()

        self.data_lock = threading.Lock()
        self.pending_vibration = False
        self._flask_thread = None

    def _setup_routes(self):
        @self.app.route("/imu", methods=["POST"])
        def imu():
            data = request.get_json()
            print("++++++++++   Received IMU POST:", data)
            with self.data_lock:
                for key in ["ax", "ay", "az","gx","gy","gz"]:
                    self.imu_data[key].append(data.get(key, 0.0))
            if self.pending_vibration:
                self.pending_vibration = False
                return "VIBRATE", 200
            return "OK", 200

        @self.app.route("/volume", methods=["POST"])
        def volume_event():
            event = request.form.get("event")
            if event == "VOLUME_UP":
                self.open_gripper_command = True
                self.close_gripper_command = False
            elif event == "VOLUME_DOWN":
                self.open_gripper_command = False
                self.close_gripper_command = True
            return "OK", 200

        @self.app.route("/vibrate_once", methods=["POST"])
        def vibrate_once():
            self.pending_vibration = True
            return "queued", 200

    def start(self):
        """Start the background server for HTTP IMU input."""
        self.running = True
        self._flask_thread = threading.Thread(
            target=lambda: self.app.run(host="127.0.0.1", port=self.port, threaded=True),
            daemon=True,
        )
        self._flask_thread.start()
        logging.info("IMUController HTTP server started.")

    def stop(self):
        """This doesn't stop Flask (Flask is hard to stop cleanly), but sets running = False."""
        self.running = False
        logging.info("IMUController stopped (Flask will remain running in thread).")

    def update(self):
        """Could be used in future for smoothing or gesture detection."""
        pass

    def get_deltas(self):
        """Return (dx, dy, dz) derived from the latest IMU window."""

        with self.data_lock:
            # Require at least one sample of each

            # print("I was called 1 -------------------------------")

            # for k in ["ax", "ay", "az", "gx", "gy", "gz"]:
            #     print(f"Buffer length for {k}: {len(self.imu_data[k])}")
            if any(len(self.imu_data[k]) == 0 for k in ["ax", "ay", "az", "gx", "gy", "gz"]):
                print("One or more buffers are empty, returning 0s.")
                return 0.0, 0.0, 0.0
            # print("I was called  2-------------------------------")
            
        

            # Smoothed averages
            ax = sum(self.imu_data["ax"]) / len(self.imu_data["ax"])
            ay = sum(self.imu_data["ay"]) / len(self.imu_data["ay"])
            az = sum(self.imu_data["az"]) / len(self.imu_data["az"])
            gx = sum(self.imu_data["gx"]) / len(self.imu_data["gx"])
            gy = sum(self.imu_data["gy"]) / len(self.imu_data["gy"])
            gz = sum(self.imu_data["gz"]) / len(self.imu_data["gz"])

        # 1) gyro‑based translations
        dx_g =  gz * self.gyro_scale * self.dt   # rot Z → +X
        dy_g =  gx * self.gyro_scale * self.dt   # rot X → +Y
        dz_g =  gy * self.gyro_scale * self.dt   # rot Y → +Z

        # 2) optional accel push (gravity‑compensated)
        dz_lin = (az - self.gravity_z) * self.acc_gain * self.dt
        dx_lin =  ax * self.acc_gain * self.dt
        dy_lin =  ay * self.acc_gain * self.dt

        dx = dx_g + dx_lin
        dy = dy_g + dy_lin
        dz = dz_g + dz_lin

        # print("******************************************************************************************************[get_deltas]", dx, dy, dz)

        # print("[get_deltas] self.imu_data", actiondict)

        return dx, dy, dz