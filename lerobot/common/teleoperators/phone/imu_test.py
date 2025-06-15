import time
import requests
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ─────────────── Config ────────────────────────────────────────────────
SERVER = "http://127.0.0.1:5010"   # adjust if remote
POLL_DT = 0.05                     # 20 Hz polling

GYRO_SCALE = 0.5                 # metres per (deg/s × dt)
ACC_GAIN   = 0.02                 # metres per (m/s² × dt)  (set 0 to disable)
GRAVITY_Z  = 9.8                   # subtract gravity from az

POS_BOUNDS = np.array([[-1, 1],    # x min/max
                       [-1, 1],    # y
                       [-1, 1]])   # z

# ─────────────── Initialise figure ─────────────────────────────────────
plt.ion()
fig = plt.figure(figsize=(6, 6))
ax3d = fig.add_subplot(111, projection='3d')
ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
ax3d.set_title("IMU‑driven Repère")

# initial pose
pos = np.array([0.2, 0.0, 0.3], dtype=float)
quivers = []

def draw_repere(p):
    global quivers
    # clear previous quivers
    for q in quivers: q.remove()
    quivers.clear()

    ax3d.set_xlim(*POS_BOUNDS[0]); ax3d.set_ylim(*POS_BOUNDS[1]); ax3d.set_zlim(*POS_BOUNDS[2])

    axes = np.eye(3) * 0.1  # 10 cm arrows
    colors = ['r', 'g', 'b']
    for vec, c in zip(axes, colors):
        q = ax3d.quiver(p[0], p[1], p[2], vec[0], vec[1], vec[2], color=c, linewidth=3)
        quivers.append(q)

    plt.draw(); plt.pause(0.001)

draw_repere(pos)
print("🚀 Viewer running (Ctrl‑C to exit)")

# ─────────────── Helper: apply gyro → translation ─────────────────────
def gyro_to_dpos(gx, gy, gz, dt):
    dx =  gz * GYRO_SCALE * dt   # rot Z → +X
    dy =  gx * GYRO_SCALE * dt   # rot X → +Y
    dz =  gy * GYRO_SCALE * dt   # rot Y → +Z
    return np.array([dx, dy, dz])

# ─────────────── Main Loop ────────────────────────────────────────────
try:
    while True:
        try:
            r = requests.get(f"{SERVER}/imu/latest", timeout=0.3)
            data = r.json()
            ax_, ay_, az_ = data.get("ax",0), data.get("ay",0), data.get("az",0)
            gx_, gy_, gz_ = data.get("gx",0), data.get("gy",0), data.get("gz",0)
        except Exception as e:
            print("⚠️  fetch failed:", e)
            time.sleep(POLL_DT); continue

        # -------- translation from gyro --------
        dpos_gyro = gyro_to_dpos(gx_, gy_, gz_, POLL_DT)

        # -------- optional accel contribution ---
        acc = np.array([ax_, ay_, az_ - GRAVITY_Z], dtype=float)
        dpos_acc = acc * ACC_GAIN * POLL_DT if ACC_GAIN else 0

        # -------- integrate & clamp ------------
        pos += dpos_gyro + dpos_acc
        pos = np.clip(pos, POS_BOUNDS[:,0], POS_BOUNDS[:,1])

        draw_repere(pos)
        time.sleep(POLL_DT)

except KeyboardInterrupt:
    print("\nViewer stopped.")