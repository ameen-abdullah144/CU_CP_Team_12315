import cv2
import time
import threading
from flask import Flask, Response, render_template, jsonify
from config import DASHBOARD_HOST, DASHBOARD_PORT, STREAM_FPS

app = Flask(__name__)


class SharedState:
    frame         = None
    vehicle_count = 0
    alarms        = {}
    recent_events = []
    lock          = threading.Lock()


state = SharedState()


def update_state(frame, vehicle_count, alarms, recent_events):
    with state.lock:
        state.frame         = frame.copy()
        state.vehicle_count = vehicle_count
        state.alarms        = alarms
        state.recent_events = recent_events


def generate_stream():
    """MJPEG stream — browser receives this as a live video feed."""
    while True:
        with state.lock:
            frame = state.frame
        if frame is None:
            time.sleep(0.05)
            continue
        ret, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
        )
        if not ret:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )
        time.sleep(1 / STREAM_FPS)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/status")
def api_status():
    with state.lock:
        return jsonify({
            "vehicle_count": state.vehicle_count,
            "alarms":        state.alarms,
            "events":        state.recent_events
        })


def run_dashboard():
    app.run(
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        debug=False,
        threaded=True
    )