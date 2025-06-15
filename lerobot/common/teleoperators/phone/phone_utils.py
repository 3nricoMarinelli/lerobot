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
    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01, port=5010):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.port = port

        # Buffers for incoming IMU data
        self.N = 5
        self.imu_data = {
            "ax": deque(maxlen=self.N),
            "ay": deque(maxlen=self.N),
            "az": deque(maxlen=self.N),
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
            with self.data_lock:
                for key in ["ax", "ay", "az"]:
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
            target=lambda: self.app.run(host="0.0.0.0", port=self.port, threaded=True),
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
        """Convert IMU acceleration data to dx, dy, dz values."""
        with self.data_lock:
            if not all(len(self.imu_data[k]) > 0 for k in ["ax", "ay", "az"]):
                return 0.0, 0.0, 0.0

            # Use average to reduce noise
            avg_ax = sum(self.imu_data["ax"]) / len(self.imu_data["ax"])
            avg_ay = sum(self.imu_data["ay"]) / len(self.imu_data["ay"])
            avg_az = sum(self.imu_data["az"]) / len(self.imu_data["az"])

        dx = avg_ax * self.x_step_size
        dy = avg_ay * self.y_step_size
        dz = avg_az * self.z_step_size
        return dx, dy, dz

# from flask import Flask, request
# import threading
# import time
# import matplotlib.pyplot as plt
# from collections import deque

# app = Flask(__name__)
# pending_vibration = False

# # Buffers for IMU data (last N points)
# N = 100
# imu_data = {
#     'ax': deque(maxlen=N),
#     'ay': deque(maxlen=N),
#     'az': deque(maxlen=N),
#     'gx': deque(maxlen=N),
#     'gy': deque(maxlen=N),
#     'gz': deque(maxlen=N)
# }

# # Lock to prevent race conditions
# data_lock = threading.Lock()

# @app.route('/volume', methods=['POST'])
# def volume_event():
#     event = request.form.get('event')
#     print(f"Received volume event: {event}")
#     if event == "VOLUME_UP":
#         print("Move servoR to grip")
#     elif event == "VOLUME_DOWN":
#         print("Release servo grip")
#     return 'OK', 200

# @app.route('/imu', methods=['POST'])
# def imu():
#     global pending_vibration
#     data = request.get_json()
#     print("IMU:", data)

#     with data_lock:
#         for key in imu_data:
#             imu_data[key].append(data.get(key, 0))

#     if pending_vibration:
#         pending_vibration = False
#         return "VIBRATE", 200
#     return "OK", 200

# @app.route('/vibrate_once', methods=['POST'])
# def vibrate_once():
#     global pending_vibration
#     pending_vibration = True
#     return 'queued', 200

# def plot_thread():
#     plt.ion()
#     fig, axs = plt.subplots(2, 1, figsize=(10, 6))

#     while True:
#         with data_lock:
#             axs[0].cla()
#             axs[1].cla()
#             axs[0].set_title("Accelerometer (ax, ay, az)")
#             axs[1].set_title("Gyroscope (gx, gy, gz)")

#             axs[0].plot(imu_data['ax'], label='ax')
#             axs[0].plot(imu_data['ay'], label='ay')
#             axs[0].plot(imu_data['az'], label='az')

#             axs[1].plot(imu_data['gx'], label='gx')
#             axs[1].plot(imu_data['gy'], label='gy')
#             axs[1].plot(imu_data['gz'], label='gz')

#             axs[0].legend()
#             axs[1].legend()

#         plt.pause(0.05)

# if __name__ == "__main__":
#     # Start plotting in a background thread
#     threading.Thread(target=plot_thread, daemon=True).start()

#     # Start Flask app
#     app.run(host='127.0.0.1', port=5010)
