import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def normalize(v):
    return v / np.linalg.norm(v)

def get_hand_euler_xyz(lm):
    # 3 points pour définir la paume
    p0 = np.array([lm.landmark[0].x, lm.landmark[0].y, lm.landmark[0].z])
    p5 = np.array([lm.landmark[5].x, lm.landmark[5].y, lm.landmark[5].z])
    p17 = np.array([lm.landmark[17].x, lm.landmark[17].y, lm.landmark[17].z])

    # axes locaux
    x_axis = normalize(p5 - p0)
    y_axis = normalize(p17 - p0)
    z_axis = normalize(np.cross(x_axis, y_axis))

    # matrice de rotation R = [x y z]
    Rm = np.column_stack((x_axis, y_axis, z_axis))

    # extraire roll (X), pitch (Y), yaw (Z) en radian
    roll  = math.atan2( Rm[2,1],  Rm[2,2])
    pitch = math.atan2(-Rm[2,0], math.hypot(Rm[2,1], Rm[2,2]))
    yaw   = math.atan2( Rm[1,0],  Rm[0,0])

    # en degrés
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

cap = cv2.VideoCapture(0)
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
            roll, pitch, yaw = get_hand_euler_xyz(lm)
            cv2.putText(img,
                        f"X:{roll:.1f}  Y:{pitch:.1f}  Z:{yaw:.1f}",
                        (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2)
            mp_drawing.draw_landmarks(
                img, lm, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

        cv2.imshow("Hand → X Y Z (°)", img)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
