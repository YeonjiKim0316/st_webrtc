import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import cv2
import asyncio

# asyncio 이벤트 루프 정책 초기화
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# YOLOv8 모델 로드
model = YOLO("yolov8n.pt")  # YOLO 모델 경량화 버전 사용

# Streamlit UI 설정
st.title("YOLOv8 Object Detection")
st.write("실시간 객체 감지를 위해 WebRTC를 사용합니다.")

# WebRTC 설정
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # YOLOv8 객체 감지 수행
    results = model(img)

    # 감지된 객체에 대한 경계 상자와 레이블 그리기
    if results and hasattr(results, 'boxes'):
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 클래스 이름과 신뢰도 표시
            if hasattr(box, 'cls') and hasattr(box, 'conf'):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names[cls]} {conf:.2f}"
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
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:global.stun.twilio.com:3478?transport=udp"]}
        ]
    },
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)
