# base source: https://github.com/jtaquia/open_cv_CAMERA_STREAMLIT/blob/main/Stream_CV2_Video.py

import cv2
import numpy as np
import streamlit as st
from plots import Barplot, build_hist
from src.emotion_detector import _get_emotion_model, gen_detect_emotion
from src.face_detector import _get_face_model
from src.facemask_detector import _get_facemask_model, gen_detect_facemask
from src.utils import get_webcam

st.set_page_config(layout="wide")


# Get Webcamindex
_, webcamindex = get_webcam()

# Load models
mood_model = _get_emotion_model()
facemask_model = _get_facemask_model()
facenet = _get_face_model()

### SIDEBAR PAGE

st.sidebar.header("Settings")
WEBCAMINDEX = st.sidebar.selectbox("Webcam", webcamindex)
# WEBCAMINDEX = max(webcamindex)
# WEBCAMINDEX = 1

usecase = st.sidebar.selectbox("Detection Model", ["Emotion Recognition", "Facemask Detection"])

### CENTER PAGE

st.title(f"{usecase}")


match usecase:
    case "Emotion Recognition":

        def clock_func(c, sp):
            if c == 0:
                c = sp
                return c, True
            else:
                c -= 1
                return c, False

        emrec_run = st.checkbox("Make it run!")

        camera = cv2.VideoCapture(WEBCAMINDEX)
        history = None
        step = 5
        cnt = 0
        maxidx = 0

        moods = [
            "Angry",
            "Disgusted",
            "Fearful",
            "Happy",
            "Neutral",
            "Sad",
            "Surprised",
        ]

        col1, col2 = st.columns(2)

        with col1:
            FRAME_WINDOW = st.image([])

        with col2:
            current_plot = Barplot("Current Mood", moods)

        while emrec_run:
            ### THIS IS NEW

            _, frame = camera.read()

            cnt, clc = clock_func(cnt, step)

            frame, l, maxidx = gen_detect_emotion(frame, mood_model, facenet, clc, maxidx)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            FRAME_WINDOW.image(frame)

            history, agg = build_hist(history, l, moods)

            current_plot.update(agg)

    case "Facemask Detection":
        facemask_run = st.checkbox("Make it run!")
        masks = ["No mask", "Mask"]
        camera = cv2.VideoCapture(WEBCAMINDEX)
        history = None

        col1, col2 = st.columns(2)

        with col1:
            FRAME_WINDOW = st.image([])

        with col2:
            current_plot = Barplot("Mask detected:", masks)

        while facemask_run:
            _, frame = camera.read()

            frame, pred_list = gen_detect_facemask(frame, facemask_model, facenet)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

            history, agg = build_hist(history, pred_list, masks)
            current_plot.update(agg)
    case _:
        print("Command not recognized")
