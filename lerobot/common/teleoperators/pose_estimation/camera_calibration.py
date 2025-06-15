import cv2
import numpy as np

CHECKERBOARD = (8, 6)  # nombre de coins internes (8x6)
SQUARE_SIZE = 115.0  # en mm

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

cap = cv2.VideoCapture(1)
print("🟡 Appuie sur ESPACE pour capturer un damier détecté")
print("🔵 Appuie sur 'C' pour lancer la calibration (au moins 10 images)")
print("🔴 Appuie sur 'ESC' pour quitter")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    display = frame.copy()
    if found:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners2, found)

    cv2.imshow("Calibration", display)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == 32 and found:  # ESPACE
        objpoints.append(objp.copy())
        imgpoints.append(corners2)
        print(f"✔ Image capturée : total = {len(objpoints)}")
    elif key == ord('c') and len(objpoints) >= 10:
        print("⏳ Calibration en cours...")
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("✅ Calibration réussie")
        print("Matrice intrinsèque K :\n", K)
        print("Coefficients de distorsion :\n", dist.ravel())
        # Sauvegarde avec les mêmes noms que dans le code de lecture
        np.savez("calibration_data.npz", camera_matrix=K, dist_coeffs=dist)
        print("📁 Données enregistrées dans calibration_data.npz")

cap.release()
cv2.destroyAllWindows()
