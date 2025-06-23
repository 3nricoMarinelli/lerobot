import cv2
import mediapipe as mp
import numpy as np
import math
import socket
import struct
import time



mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('localhost', 11310)

# Charger calibration
with np.load("lerobot/common/teleoperators/pose_estimation/calibration_data.npz") as X:
    K = X["camera_matrix"]
    dist = X["dist_coeffs"]

REAL_LENGTH_CM = 8.0

def undistort_landmark(landmark, K, dist, frame_shape):
    h, w = frame_shape[:2]
    pt = np.array([[landmark.x * w, landmark.y * h]], dtype=np.float32)
    pt_undist = cv2.undistortPoints(pt.reshape(-1, 1, 2), K, dist, P=K)
    return pt_undist[0,0]

def pixel_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def normalize(v):
    return v / np.linalg.norm(v)

def get_hand_axes_and_origin(lm, ref_idx=[0, 5, 17, 9]):
    p0 = np.array([lm.landmark[ref_idx[0]].x, lm.landmark[ref_idx[0]].y, lm.landmark[ref_idx[0]].z])
    p5 = np.array([lm.landmark[ref_idx[1]].x, lm.landmark[ref_idx[1]].y, lm.landmark[ref_idx[1]].z])
    p17 = np.array([lm.landmark[ref_idx[2]].x, lm.landmark[ref_idx[2]].y, lm.landmark[ref_idx[2]].z])
    p9 = np.array([lm.landmark[ref_idx[3]].x, lm.landmark[ref_idx[3]].y, lm.landmark[ref_idx[3]].z])

    x_axis = normalize(p5 - p0)
    y_axis = normalize(p17 - p0)
    z_axis = normalize(np.cross(x_axis, y_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))

    Rm = np.column_stack((x_axis, y_axis, z_axis))
    return Rm, p0, p9

def get_euler_angles(Rm):
    roll  = math.atan2(Rm[2,1], Rm[2,2])
    pitch = math.atan2(-Rm[2,0], math.hypot(Rm[2,1], Rm[2,2]))
    yaw   = math.atan2(Rm[1,0], Rm[0,0])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

def draw_axes(img, origin_lm, R, scale=100):
    h, w, _ = img.shape
    center = np.array([origin_lm.x * w, origin_lm.y * h])
    axes = R * scale

    x_end = (center + axes[:, 0][:2]).astype(int)
    y_end = (center + axes[:, 1][:2]).astype(int)
    z_end = (center + axes[:, 2][:2]).astype(int)
    center_pt = center.astype(int)

    cv2.line(img, center_pt, x_end, (0, 0, 255), 2)
    cv2.line(img, center_pt, y_end, (0, 255, 0), 2)
    cv2.line(img, center_pt, z_end, (255, 0, 0), 2)

    cv2.putText(img, 'X', tuple(x_end), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img, 'Y', tuple(y_end), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, 'Z', tuple(z_end), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

def send_hand_data(pos, euler, opening_ratio):
    data = struct.pack('7f',
                       pos[0], pos[1], pos[2],
                       euler[0], euler[1], euler[2],
                       opening_ratio)
    sock.sendto(data, server_address)

cap = cv2.VideoCapture(0)

reference_position = None
reference_rotation = None

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1
) as hands:

    prev_time = 0
    target_fps = 30
    frame_time = 1.0 / target_fps

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if now - prev_time < frame_time:
            time.sleep(frame_time - (now - prev_time))
        prev_time = time.time()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        res = hands.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]

            Rm, p0, p9 = get_hand_axes_and_origin(lm)
            draw_axes(img, lm.landmark[0], Rm)

            # Repères non déformés
            pt0 = undistort_landmark(lm.landmark[0], K, dist, img.shape)
            pt9 = undistort_landmark(lm.landmark[9], K, dist, img.shape)

            dist_px = pixel_distance(pt0, pt9)
            focal_length_px = (K[0,0] + K[1,1]) / 2
            distance_cm = (focal_length_px * REAL_LENGTH_CM) / dist_px

            x_cm = (pt0[0] - K[0,2]) * distance_cm / K[0,0]
            y_cm = (pt0[1] - K[1,2]) * distance_cm / K[1,1]
            z_cm = distance_cm
            pos_cm = np.array([x_cm, y_cm, z_cm])

            if reference_position is None:
                reference_position = pos_cm
            if reference_rotation is None:
                reference_rotation = Rm

            rel_pos_cm = pos_cm - reference_position
            rel_rot = reference_rotation.T @ Rm
            euler_angles = get_euler_angles(rel_rot)

            # Calcul de l'ouverture normalisée
            pt_thumb = undistort_landmark(lm.landmark[4], K, dist, img.shape)
            pt_pinky = undistort_landmark(lm.landmark[20], K, dist, img.shape)
            pt_index = undistort_landmark(lm.landmark[5], K, dist, img.shape)
            pt_ring = undistort_landmark(lm.landmark[17], K, dist, img.shape)

            ouverture_px = pixel_distance(pt_thumb, pt_pinky)
            largeur_paume_px = pixel_distance(pt_index, pt_ring)
            opening_ratio = ouverture_px / largeur_paume_px

            send_hand_data(rel_pos_cm, euler_angles, opening_ratio)

            roll, pitch, yaw = euler_angles
            cv2.putText(img,
                        f"X:{rel_pos_cm[0]:+.1f}cm Y:{rel_pos_cm[1]:+.1f}cm Z:{rel_pos_cm[2]:+.1f}cm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(img,
                        f"Roll:{roll:+.1f} Pitch:{pitch:+.1f} Yaw:{yaw:+.1f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(img,
                        f"Open ratio:{opening_ratio:.2f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            mp_drawing.draw_landmarks(
                img, lm, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

        cv2.imshow("pose_extract", img)

        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):
            reference_position = None
            reference_rotation = None

cap.release()
cv2.destroyAllWindows()
sock.close()
