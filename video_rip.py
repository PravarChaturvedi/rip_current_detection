import cv2
import streamlit as st
from ultralytics import YOLO

# Replace the relative path to your weight file
model_path = "yolo_50epoch.pt"

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",  # Setting page title
    page_icon="🤖",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded"    # Expanding sidebar by default
)

# Creating sidebar
with st.sidebar:
    st.header("Image/Video Config")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting videos
    source_vid = st.sidebar.selectbox(
        "Choose a video...",
        ["VID20220829120026.mp4","VID20220829121402.mp4",])

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 0, 100, 40)) / 100

# Creating main page heading
st.title("Object Detection using YOLOv8")

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
# st.write("Model loaded successfully!")

if source_vid is not None:
    with open(str(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)
    if st.sidebar.button('Detect Objects'):
        vid_cap = cv2.VideoCapture(
            source_vid)
        st_frame = st.empty()
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            if success:
                image = cv2.resize(image, (720, int(720*(9/16))))
                res = model.predict(image, conf=0.01, iou = 0.3)
                result_tensor = res[0].boxes
                res_plotted = res[0].plot()
                st_frame.image(res_plotted,
                               caption='Detected Video',
                               channels="BGR",
                               use_column_width=True
                               )
            else:
                vid_cap.release()
                break