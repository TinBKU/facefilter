import cv2
import numpy as np

def apply_filter(frame, landmarks, filter_path):
    # Tải ảnh filter
    filter_img = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED)

    # Kiểm tra nếu ảnh không có kênh alpha
    if filter_img.shape[2] != 4:
        h, w, _ = filter_img.shape
        alpha_channel = np.ones((h, w), dtype=np.uint8) * 255
        filter_img = np.dstack([filter_img, alpha_channel])

    # Lấy vị trí mắt trái và phải
    left_eye = landmarks[33]  # Mắt trái
    right_eye = landmarks[263]  # Mắt phải

    # Tính toán kích thước và vị trí filter
    width = int(np.linalg.norm(np.array(left_eye) - np.array(right_eye)) * 1.5)
    height = int(width / filter_img.shape[1] * filter_img.shape[0])
    x = max(0, int((left_eye[0] + right_eye[0]) / 2 - width / 2))
    y = max(0, int(left_eye[1] - height / 2))

    # Điều chỉnh để filter không vượt quá kích thước ảnh
    frame_height, frame_width, _ = frame.shape
    width = min(width, frame_width - x)
    height = min(height, frame_height - y)

    filter_resized = cv2.resize(filter_img, (width, height))

    # Tách kênh alpha
    alpha_mask = filter_resized[:, :, 3] / 255.0

    # Áp dụng filter lên ảnh gốc
    for c in range(3):
        frame[y:y+height, x:x+width, c] = (
            frame[y:y+height, x:x+width, c] * (1 - alpha_mask)
            + filter_resized[:, :, c] * alpha_mask
        )

    return frame
