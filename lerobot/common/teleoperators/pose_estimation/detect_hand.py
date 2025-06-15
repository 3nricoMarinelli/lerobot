import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

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
    y_axis = normalize(np.cross(z_axis, x_axis))  # orthogonalisation

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

cap = cv2.VideoCapture(1)

reference_rotation = None
reference_position = None
reference_scale_z = None

def to_mm(norm_coord, frame_size):
    return norm_coord * frame_size * 200

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        res = hands.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            Rm, p0, p9 = get_hand_axes_and_origin(lm)

            draw_axes(img, lm.landmark[0], Rm)

            frame_h, frame_w, _ = img.shape

            # Calcul du scale Z par la distance p0-p9 en pixels (distance réelle inconnue, mais constante)
            p0_px = np.array([lm.landmark[0].x * frame_w, lm.landmark[0].y * frame_h])
            p9_px = np.array([lm.landmark[9].x * frame_w, lm.landmark[9].y * frame_h])
            scale_z = np.linalg.norm(p0_px - p9_px)

            # Approximate position X Y Z en mm (X, Y normalisés convertis, Z estimé par scale inverse)
            pos_x = to_mm(p0[0], frame_w)
            pos_y = to_mm(p0[1], frame_h)

            if reference_scale_z is None:
                reference_scale_z = scale_z

            # Inversion: plus la main est proche, plus scale_z est grand => Z diminue quand scale_z augmente
            # On fait un Z relatif en mm, proportionnel à l'inverse du scale_z
            pos_z = 200 * (reference_scale_z / scale_z - 1)  # Z=0 au départ

            pos_mm = np.array([pos_x, pos_y, pos_z])

            if reference_position is None:
                reference_position = pos_mm
            if reference_rotation is None:
                reference_rotation = Rm

            rel_pos = pos_mm - reference_position
            rel_rot = reference_rotation.T @ Rm
            roll, pitch, yaw = get_euler_angles(rel_rot)

            text_pos = f"X: {rel_pos[0]:+.1f}mm  Y: {rel_pos[1]:+.1f}mm  Z: {rel_pos[2]:+.1f}mm"
            text_rot = f"Roll: {roll:+.1f}°  Pitch: {pitch:+.1f}°  Yaw: {yaw:+.1f}°"
            cv2.putText(img, text_pos, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img, text_rot, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            mp_drawing.draw_landmarks(
                img, lm, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

        cv2.imshow("Main + Repère + Position + Rotation", img)

        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):
            reference_position = None
            reference_rotation = None
            reference_scale_z = None

cap.release()
cv2.destroyAllWindows()
