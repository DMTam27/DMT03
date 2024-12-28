import cv2
import numpy as np
import requests

# URL của ESP32-CAM
url = "http://192.168.0.112/capture"

# Đường dẫn đến các tệp YOLO
weights_path = "yolov4-tiny.weights"  # Thay bằng yolov3-tiny.weights nếu dùng YOLOv3
config_path = "yolov4-tiny.cfg"       # Thay bằng yolov3-tiny.cfg nếu dùng YOLOv3
names_path = "coco.names"

# Tải các lớp (classes) từ tệp coco.names
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Tải mô hình YOLO
net = cv2.dnn.readNet(weights_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Lấy thông tin các lớp đầu ra của YOLO
layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def fetch_image(url):
    """Lấy hình ảnh từ ESP32-CAM thông qua HTTP."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Không thể lấy hình ảnh từ ESP32-CAM: {e}")
        return None

def detect_objects(frame):
    """Nhận diện các đồ vật trong khung hình."""
    height, width, _ = frame.shape
    
    # Tạo blob từ khung hình
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Chạy mô hình YOLO để lấy các dự đoán
    layer_outputs = net.forward(out_layers)
    
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:  # Ngưỡng phát hiện
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Lọc các hộp (boxes) bằng Non-Maximum Suppression (NMS)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            
            # Vẽ hộp và hiển thị nhãn
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

while True:
    frame = fetch_image(url)
    if frame is not None:
        detected_frame = detect_objects(frame)
        cv2.imshow("Object Detection", detected_frame)
    else:
        print("Không nhận được hình ảnh, thử lại...")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

