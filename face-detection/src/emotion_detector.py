import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def mean(a):
    return sum(a) / len(a)


def _get_emotion_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation="softmax"))
    model.load_weights("face-detection/models/emotionmodel.h5")
    model.compile()

    return model


def gen_detect_emotion(frame, mood_model, facenet, clc, maxindex):
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {
        0: "Angry",
        1: "Disgusted",
        2: "Fearful",
        3: "Happy",
        4: "Neutral",
        5: "Sad",
        6: "Surprised",
    }
    all_prediction_list = []

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    facenet.setInput(blob)
    faces = facenet.forward()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence < 0.5:
            continue
        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x, y, x1, y1) = box.astype("int")

        if clc:
            roi_gray = gray[y:y1, x:x1]
            try:
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            except:
                continue
            # convert numpy array to tensor for faster execution with @tf.function
            cropped_img = tf.convert_to_tensor(cropped_img, dtype=tf.float32)
            prediction = mood_model.predict(cropped_img)
            # scale percentages to 100
            percentage_prediction = prediction * 100

            prediction_list = percentage_prediction.reshape(-1).tolist()
            all_prediction_list.append(prediction_list)

            maxindex = int(np.argmax(prediction))

        cv2.putText(
            frame,
            emotion_dict[maxindex],
            (x + 20, y - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (43, 121, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 0), 2)

    meaned_list = [*map(mean, zip(*all_prediction_list))]

    return frame, meaned_list, maxindex
