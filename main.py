import cv2
from utils.face_detection import detect_faces
from utils.landmarks import get_landmarks
from utils.overlay import apply_filter
import time
import openpyxl
from datetime import datetime

# Đường dẫn tới filter
sunglasses_path = 'filters/sunglasses.png'

# Khởi động camera
cap = cv2.VideoCapture(0)
# Đếm số khung hình và thời gian bắt đầu
frame_count = 0
start_time = time.time()
start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

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
    # Tăng số lượng khung hình
    frame_count += 1

    # Hiển thị kết quả
    cv2.imshow('Face Filter App', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tính FPS
end_time = time.time()
end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
fps = frame_count / (end_time - start_time)
print(f"Tốc độ xử lý: {fps:.2f} FPS")

cap.release()
cv2.destroyAllWindows()

# Lưu kết quả FPS vào file Excel
excel_file = 'fps_results.xlsx'

# Kiểm tra xem file Excel đã tồn tại chưa
try:
    wb = openpyxl.load_workbook(excel_file)
    sheet = wb.active
except FileNotFoundError:
    # Nếu chưa tồn tại, tạo file Excel mới
    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.title = 'FPS Results'
    # Tạo tiêu đề cột
    sheet.append(['Start Time', 'End Time', 'Frame Count', 'FPS'])

# Ghi kết quả vào file Excel
sheet.append([start_time_str, end_time_str, frame_count, round(fps, 2)])

# Lưu file Excel
wb.save(excel_file)
print(f"Kết quả FPS đã được lưu vào file: {excel_file}")
