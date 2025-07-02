import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'yolov5/runs/train/exp7/weights/best.pt', force_reload=True)  # lightweight YOLO
model.conf = 0.08 # reduce confidence to 25%
cap = cv2.VideoCapture(0)  # Real-time webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))  # Resize for speed
    results = model(frame)


    for *xyxy, conf, cls in results.xyxy[0]:
        class_id = int(cls)

        x1, y1, x2, y2 = map(int, xyxy)

        label = model.names[class_id] #gets class name

        confidence = float(conf) * 100 #converts confidence to percentages

        display_text = f"{label}: {confidence: .1f}%" #Text

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            

    cv2.imshow("YOLOv5 Real-Time", frame)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
