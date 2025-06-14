from flask import Flask, request
import threading
import time
import matplotlib.pyplot as plt
from collections import deque

app = Flask(__name__)
pending_vibration = False

# Buffers for IMU data (last N points)
N = 100
imu_data = {
    'ax': deque(maxlen=N),
    'ay': deque(maxlen=N),
    'az': deque(maxlen=N),
    'gx': deque(maxlen=N),
    'gy': deque(maxlen=N),
    'gz': deque(maxlen=N)
}

# Lock to prevent race conditions
data_lock = threading.Lock()

@app.route('/volume', methods=['POST'])
def volume_event():
    event = request.form.get('event')
    print(f"Received volume event: {event}")
    if event == "VOLUME_UP":
        print("Move servoR to grip")
    elif event == "VOLUME_DOWN":
        print("Release servo grip")
    return 'OK', 200

@app.route('/imu', methods=['POST'])
def imu():
    global pending_vibration
    data = request.get_json()
    print("IMU:", data)

    with data_lock:
        for key in imu_data:
            imu_data[key].append(data.get(key, 0))

    if pending_vibration:
        pending_vibration = False
        return "VIBRATE", 200
    return "OK", 200

@app.route('/vibrate_once', methods=['POST'])
def vibrate_once():
    global pending_vibration
    pending_vibration = True
    return 'queued', 200

def plot_thread():
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    while True:
        with data_lock:
            axs[0].cla()
            axs[1].cla()
            axs[0].set_title("Accelerometer (ax, ay, az)")
            axs[1].set_title("Gyroscope (gx, gy, gz)")

            axs[0].plot(imu_data['ax'], label='ax')
            axs[0].plot(imu_data['ay'], label='ay')
            axs[0].plot(imu_data['az'], label='az')

            axs[1].plot(imu_data['gx'], label='gx')
            axs[1].plot(imu_data['gy'], label='gy')
            axs[1].plot(imu_data['gz'], label='gz')

            axs[0].legend()
            axs[1].legend()

        plt.pause(0.05)

if __name__ == "__main__":
    # Start plotting in a background thread
    threading.Thread(target=plot_thread, daemon=True).start()

    # Start Flask app
    app.run(host='127.0.0.1', port=5010)
