# camera_control.py

import socket
import struct
from typing import Any

from ..teleoperator import Teleoperator

class CameraControl(Teleoperator):
    """
    Controller that receives delta translation (Δx, Δy, Δz) over UDP
    from a MediaPipe-based vision system and sends Cartesian motion commands.
    """

    name = "camera_control"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.sock = None

    def connect(self) -> None:
        """Bind to the UDP socket receiving hand tracking deltas."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('localhost', 11310))
        self.sock.settimeout(0.01)  # Non-blocking behavior

    def disconnect(self) -> None:
        if self.sock:
            self.sock.close()
            self.sock = None

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (3,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    def get_action(self) -> dict[str, Any]:
        if not self.sock:
            raise RuntimeError("CameraControl not connected")

        try:
            data, _ = self.sock.recvfrom(24)  # 6 floats = 24 bytes
            position = struct.unpack('6f', data)[0:3]
            dx, dy, dz = position
        except (socket.timeout, struct.error):
            dx = dy = dz = 0.0

        return {
            "delta_x": float(dx),
            "delta_y": float(dy),
            "delta_z": float(dz),
        }

    def is_connected(self) -> bool:
        return self.sock is not None

    def calibrate(self)        -> None: pass
    def is_calibrated(self)    -> bool: return True
    def configure(self)        -> None: pass
    def send_feedback(self, *_): pass
