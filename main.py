import cv2
from utils.face_detection import detect_faces
from utils.landmarks import get_landmarks
from utils.overlay import apply_filter

# Đường dẫn tới filter
sunglasses_path = 'filters/sunglasses.png'

# Khởi động camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện khuôn mặt
    faces = detect_faces(frame)

    for face in faces:
        # Lấy landmarks của từng khuôn mặt
        landmarks = get_landmarks(frame, face)

        if landmarks:
            # Áp dụng filter kính râm
            frame = apply_filter(frame, landmarks, sunglasses_path)

    # Hiển thị kết quả
    cv2.imshow('Face Filter App', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
