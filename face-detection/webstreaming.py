import argparse
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, render_template
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from src.emotion_detector import _get_emotion_model, gen_detect_emotion

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)

# initialize the video stream and
vs = cv2.VideoCapture(1)
if not vs.isOpened():
    print("Cannot open camera")
    exit()

# allow the camera sensor to warmup
time.sleep(2.0)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def frame_with_plot(frame, hist_data):
    frame_height, frame_width, _ = frame.shape
    fig = Figure(figsize=(frame_width / 100, frame_height / 100), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_xlabel("emotions in frame")
    ax.bar(*hist_data)
    canvas.draw()  # draw the canvas, cache the renderer
    pixel_width, pixel_height = fig.get_size_inches() * fig.get_dpi()
    plot_image = np.fromstring(canvas.tostring_rgb(), dtype="uint8")
    plot_image = plot_image.reshape(int(pixel_height), int(pixel_width), 3)
    frame_and_plot = np.concatenate((frame, plot_image), axis=1)
    return frame_and_plot


def emotions_in_frame(frame, mean_emotion):
    # mean_emotion = np.random.random_sample((7,))
    return frame_with_plot(
        frame,
        (
            ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"],
            mean_emotion,
        ),
    )


def detect_emotion():
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock

    print(3)

    first = True
    # loop over frames from the video stream

    while True:
        # read the next frame from the video stream, resize it
        ret, frame = vs.read()
        outputFrame_ = None
        try:
            outputFrame_, l = gen_detect_emotion(frame)
            # only plot histogram if emotions were detected
            if l:
                outputFrame_ = emotions_in_frame(outputFrame_, l)
        except cv2.error:
            outputFrame_ = frame.copy()

        with lock:
            outputFrame = outputFrame_.copy()


def generate():
    # grab global references to the output frame and lock variables
    print(2)

    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n")


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    print(1)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == "__main__":
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--ip",
        type=str,
        required=False,
        help="ip address of the device",
        default="127.0.0.1",
    )
    ap.add_argument(
        "-o",
        "--port",
        type=int,
        required=False,
        help="ephemeral port number of the server (1024 to 65535)",
        default=8000,
    )
    args = vars(ap.parse_args())
    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_emotion)
    t.daemon = True
    t.start()
    # start the flask app
    app.run(
        host=args["ip"],
        port=args["port"],
        debug=True,
        threaded=True,
        use_reloader=False,
    )
# release the video stream pointer
vs.release()
