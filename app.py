import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import cv2
import asyncio
import nest_asyncio

# asyncio 이벤트 루프 정책 초기화 및 기존 루프 처리
try:
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    nest_asyncio.apply()
except RuntimeError:
    pass

# Streamlit UI 설정
st.title("YOLOv8 Object Detection")
st.write("실시간 객체 감지를 위해 WebRTC를 사용합니다.")

# 사이드바 UI 추가
default_model = "yolov8n.pt"
st.sidebar.title("설정")
model_type = st.sidebar.selectbox("YOLO 모델 선택", ["yolov8n", "yolov8m", "yolov8l"], index=0)
confidence_threshold = st.sidebar.slider("신뢰도 임계값", 0.0, 1.0, 0.5)

# YOLOv8 모델 로드
model_path = f"{model_type}.pt"
st.sidebar.write(f"사용 중인 모델: {model_path}")
model = YOLO(model_path)

# WebRTC 설정
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # YOLOv8 객체 감지 수행
    results = model(img)

    # 감지된 객체에 대한 경계 상자와 레이블 그리기
    for result in results:
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0])

                # 신뢰도 임계값 조건 확인
                if conf < confidence_threshold:
                    continue

                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"

                # 경계 상자와 레이블 그리기
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="yolov8-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]}
        ]
    },
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)

st.sidebar.write("WebRTC를 시작하려면 위의 카메라 권한을 허용하세요.")


from ultralytics import solutions

inf = solutions.Inference(
    model="yolo11n.pt",  # You can use any model that Ultralytics support, i.e. YOLO11, YOLOv10
)

inf.inference()

### Make sure to run the file using command `streamlit run <file-name.py>`
