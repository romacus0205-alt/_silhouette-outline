"""
실루엣 아웃라인 웹앱 (YOLOv8-Seg 기반) - Streamlit 버전

Streamlit Community Cloud(완전 무료)에 배포하기 위해
Gradio 대신 Streamlit + streamlit-webrtc로 작성했습니다.

- 웹캠 캡처는 streamlit-webrtc가 브라우저(WebRTC)에서 처리하고,
  각 프레임을 서버(VideoProcessor.recv)로 보내 YOLO 추론 후 다시 돌려줍니다.
- 트랙바(Confidence %, Curve Smooth) -> st.slider로 대체
"""

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO

YOLO_MODEL_NAME = "yolov8s-seg.pt"  # nano보다 정확하지만 무료 서버에서 더 느릴 수 있음

# --- 속도 개선 설정 ---
PROCESS_EVERY_N_FRAMES = 3   # 3프레임마다 1번만 YOLO 추론 (나머지는 직전 결과 재사용)
INFER_IMG_SIZE = 384         # YOLO에 넣는 이미지 크기 (기본 640 -> 384로 축소, 화면 해상도는 그대로)

OUTLINE_COLOR = (0, 255, 0)      # BGR
OUTLINE_BORDER = (0, 0, 0)
INNER_THICKNESS = 1
BORDER_THICKNESS = 2

st.set_page_config(page_title="실루엣 아웃라인", layout="centered")
st.title("실루엣 아웃라인 (YOLOv8-Seg)")
st.caption("웹캠 접근을 허용하면 실시간으로 처리됩니다. 무료 서버라 첫 로딩과 처리 속도가 느릴 수 있어요.")


@st.cache_resource
def load_model():
    return YOLO(YOLO_MODEL_NAME)


model = load_model()

conf_percent = st.slider("Confidence %", 1, 100, 50, 1)
curve_smooth = st.slider("Curve Smooth", 3, 25, 9, 2)


def smooth_closed_contour(points, window):
    if window < 3 or len(points) < window:
        return points.astype(np.int32)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.vstack([points[-pad:], points, points[:pad]])
    kernel = np.ones(window) / window
    x_smooth = np.convolve(padded[:, 0], kernel, mode="valid")
    y_smooth = np.convolve(padded[:, 1], kernel, mode="valid")
    return np.stack([x_smooth, y_smooth], axis=1).astype(np.int32)


class SilhouetteProcessor(VideoProcessorBase):
    def __init__(self):
        self.conf_percent = 50
        self.curve_smooth = 9
        self.frame_count = 0
        self.last_overlays = []  # 직전 추론 결과(그릴 항목들) 캐시

    def _compute_overlays(self, img):
        """YOLO 추론을 돌려서 그릴 항목 리스트를 만든다."""
        conf_threshold = max(int(self.conf_percent), 1) / 100.0
        results = model.predict(source=img, conf=conf_threshold,
                                 imgsz=INFER_IMG_SIZE, verbose=False)

        overlays = []
        if len(results) > 0 and results[0].masks is not None:
            result = results[0]
            names = result.names
            mask_polygons = result.masks.xy

            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = names[cls_id]

                polygon = mask_polygons[i]
                if polygon is None or len(polygon) < 3:
                    continue

                smoothed = smooth_closed_contour(polygon, int(self.curve_smooth))
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                text = f"{label} {conf * 100:.0f}%"
                overlays.append((smoothed, text, x1, y1))

        return overlays

    def _draw_overlays(self, display, overlays):
        for smoothed, text, x1, y1 in overlays:
            cv2.polylines(display, [smoothed], isClosed=True,
                          color=OUTLINE_BORDER, thickness=BORDER_THICKNESS,
                          lineType=cv2.LINE_AA)
            cv2.polylines(display, [smoothed], isClosed=True,
                          color=OUTLINE_COLOR, thickness=INNER_THICKNESS,
                          lineType=cv2.LINE_AA)

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = max(0, y1 - 8)
            cv2.rectangle(display, (x1, ty - th - 6), (x1 + tw + 6, ty + 4),
                          (0, 0, 0), -1)
            cv2.putText(display, text, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, OUTLINE_COLOR, 2, cv2.LINE_AA)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        display = img.copy()

        self.frame_count += 1
        if self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
            self.last_overlays = self._compute_overlays(img)

        self._draw_overlays(display, self.last_overlays)

        return av.VideoFrame.from_ndarray(display, format="bgr24")


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

ctx = webrtc_streamer(
    key="silhouette-outline",
    video_processor_factory=SilhouetteProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

# 슬라이더 값을 실행 중인 VideoProcessor에 실시간으로 반영
if ctx.video_processor:
    ctx.video_processor.conf_percent = conf_percent
    ctx.video_processor.curve_smooth = curve_smooth
