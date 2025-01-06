import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

def get_landmarks(frame, face):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []
            for lm in face_landmarks.landmark:
                ih, iw, _ = frame.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                landmarks.append((x, y))
            return landmarks
    return None
