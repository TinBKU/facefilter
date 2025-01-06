import cv2
import mediapipe as mp

# Khởi tạo Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def detect_faces(frame):
    """
    Phát hiện khuôn mặt trong khung hình.
    :param frame: Ảnh đầu vào
    :return: Danh sách tọa độ khuôn mặt
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            faces.append((x, y, w, h))
    return faces
