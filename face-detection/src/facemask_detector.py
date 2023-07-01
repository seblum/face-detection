import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def _get_facemask_model():
    return load_model("face-detection/models/facemodel.h5")


def gen_detect_facemask(frame, facemask_model, facenet):
    type(frame)
    type(facemask_model)
    type(facenet)
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
    # dictionary which assigns each label an emotion (alphabetical order)
    mask_dict = {0: "no mask", 1: "mask"}
    pred_list = [[0, 0]]

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    facenet.setInput(blob)
    faces = facenet.forward()

    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence < 0.5:
            continue
        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x, y, x1, y1) = box.astype("int")

        img_crop = frame[y:y1, x:x1, :]

        try:
            img = np.expand_dims(cv2.resize(img_crop, (160, 160)), 0)
        except:
            continue
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)

        prediction = facemask_model.predict(img)
        percentage_prediction = prediction * 100

        pred_list = np.round(percentage_prediction, 2)
        # add mask to frame
        maxindex = int(prediction.argmax(axis=1))
        label = f"{mask_dict[maxindex]}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        if maxindex == 0:
            color = (0, 0, 255)  # red in BGR
        else:
            color = (0, 255, 0)
        cv2.putText(frame, label, (x + 20, y - 60), font, 1, color, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x1, y1), color, 2)

        type(frame)
        type(pred_list)
    return frame, pred_list[0]
