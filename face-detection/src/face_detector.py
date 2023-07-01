import cv2

modelFile = "face-detection/models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "face-detection/models/deploy.prototxt.txt"


def _get_face_model():
    return cv2.dnn.readNetFromCaffe(configFile, modelFile)
