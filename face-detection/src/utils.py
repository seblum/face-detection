from typing import Tuple

import cv2
import streamlit as st


# Check what index source your video comes from
def get_webcam() -> Tuple[dict, list]:
    webcam_dict = dict()
    valid_list = []
    for i in range(0, 10):
        cap = cv2.VideoCapture(i)
        is_camera = cap.isOpened()
        if is_camera:
            webcam_dict[f"index[{i}]"] = "VALID"
            cap.release()
            valid_list.append(i)
        else:
            webcam_dict[f"index[{i}]"] = None
    return webcam_dict, valid_list


if __name__ == "__main__":
    st.title("WebCam index validation check")
    webcam_dict, _ = get_webcam()
    st.write(webcam_dict)
