import time
import cv2
import requests
from ultralytics import YOLO

MODEL_PATH = r"C:\Users\lab4dx\Desktop\CAThinQ\CAThinQ\runs\detect\train\weights\best.pt"
RTSP_URL = "rtsp://chooare:a99999999@192.168.137.32:554/stream2"

ALERT_API_URL = "http://192.168.137.1:5000/api/events/vomit"
CAT_POSITION_API_URL = "http://192.168.137.1:5000/api/events/cat-position"

CONF_THRESHOLD = 0.6
COOLDOWN_SECONDS = 10
CAT_POSITION_INTERVAL = 0.5

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("RTSP 열기 실패")
    exit()

print("RTSP 연결 성공")

last_alert_time = 0
last_cat_position_time = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임 읽기 실패")
        time.sleep(1)
        continue

    results = model.predict(
        source=frame,
        imgsz=640,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    result = results[0]
    annotated_frame = result.plot()

    now = time.time()

    best_cat_box = None
    best_cat_conf = 0

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = result.names[cls_id]

        # 구토 알림
        if class_name == "vomit" and conf >= CONF_THRESHOLD:
            if now - last_alert_time >= COOLDOWN_SECONDS:
                payload = {
                    "type": "vomit",
                    "message": "고양이 구토가 감지되었습니다.",
                    "confidence": round(conf, 3)
                }

                try:
                    res = requests.post(ALERT_API_URL, json=payload, timeout=3)
                    print("알림 전송 완료:", res.status_code, payload)
                    last_alert_time = now
                except Exception as e:
                    print("알림 전송 실패:", e)

        # 고양이 위치 전송용: 가장 confidence 높은 cat 박스 선택
        if class_name == "cat_doll" and conf >= CONF_THRESHOLD:
            if conf > best_cat_conf:
                best_cat_conf = conf
                best_cat_box = box

    # 고양이 위치 전송
    if best_cat_box is not None and now - last_cat_position_time >= CAT_POSITION_INTERVAL:
        x_center = float(best_cat_box.xywhn[0][0])
        y_center = float(best_cat_box.xywhn[0][1])

        payload = {
            "x": round(x_center, 4),
            "y": round(y_center, 4),
            "confidence": round(best_cat_conf, 3)
        }

        try:
            requests.post(CAT_POSITION_API_URL, json=payload, timeout=1)
            print("고양이 위치 전송:", payload)
        except Exception:
            pass

        last_cat_position_time = now

    cv2.imshow("CAThinQ Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()