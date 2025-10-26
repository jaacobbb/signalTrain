import cv2
import mediapipe as mp
import numpy as np
import os

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
VIDEO_DIR = "data/raw_videos"
OUT_DIR = "data/keypoints"
os.makedirs(OUT_DIR, exist_ok=True)
for file in os.listdir(VIDEO_DIR):
    if not file.endswith(".mp4"):
        continue
    path = os.path.join(VIDEO_DIR, file)
    cap = cv2.VideoCapture(path)
    sequence = []

    with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            if results.left_hand_landmarks and results.right_hand_landmarks:
                left = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
                right = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
                pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
                full_frame = np.concatenate([pose, left, right])
            else:
                full_frame = np.zeros(33*3 + 21*3*2)

            sequence.append(full_frame)

    cap.release()
    keypoints = np.array(sequence)
    np.save(os.path.join(OUT_DIR, file.replace(".mp4", ".npy")), keypoints)
    print(f"Saved: {file.replace('.mp4', '.npy')} ({keypoints.shape})")
