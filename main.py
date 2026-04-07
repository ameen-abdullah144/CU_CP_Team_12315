import cv2
import threading
import time
import numpy as np

from config        import (CAMERA_URL, SHOW_LOCAL_WINDOW,
                            SHOW_CONFIDENCE, MIN_DISPLAY_CONF)
from detector      import VehicleDetector
from tracker       import VehicleTracker
from alarm_manager import AlarmManager
from logger        import EventLogger
from dashboard     import run_dashboard, update_state

COLOUR_NORMAL  = (0, 220, 0)
COLOUR_STOPPED = (0, 0, 255)
COLOUR_TEXT    = (255, 255, 255)


def draw_detections(frame, detections, tracked, stopped_ids):
    display_count = 0

    for det in detections:
        x, y, w, h = det["box"]
        conf        = det["confidence"]

        # Skip low-confidence boxes — don't draw them at all
        if conf < MIN_DISPLAY_CONF:
            continue

        display_count += 1
        cx, cy = x + w // 2, y + h // 2

        # ── Decide colour based on stopped status ──────────────────────────
        colour = COLOUR_NORMAL
        locked_label = det["label"]

        for tid, info in tracked.items():
            tx, ty = info["centroid"]
            if abs(tx - cx) < 40 and abs(ty - cy) < 40:
                locked_label = info.get("label", det["label"])
                if tid in stopped_ids:
                    colour = COLOUR_STOPPED
                break

        # ── Draw bounding box ──────────────────────────────────────────────
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

        # ── Label with coloured background pill ───────────────────────────
        text = f"{locked_label} {conf:.0%}" if SHOW_CONFIDENCE else locked_label
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), colour, -1)
        cv2.putText(
            frame, text,
            (x + 3, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, COLOUR_TEXT, 1
        )

    # ── Vehicle count overlay (top-left) ──────────────────────────────────
    cv2.rectangle(frame, (0, 0), (185, 36), (15, 15, 15), -1)
    cv2.putText(
        frame, f"Vehicles: {display_count}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, COLOUR_TEXT, 2
    )

    return frame, display_count


def main():
    logger   = EventLogger()
    detector = VehicleDetector()
    tracker  = VehicleTracker()
    alarms   = AlarmManager(logger)

    # ── Start Flask dashboard in background thread ─────────────────────────────
    dash_thread = threading.Thread(target=run_dashboard, daemon=True)
    dash_thread.start()
    print("[Dashboard] Running at http://localhost:5000")

    # ── Open camera or video file ──────────────────────────────────────────────
    print(f"[Camera] Connecting to: {CAMERA_URL}")
    cap = cv2.VideoCapture(CAMERA_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera/video. Check CAMERA_URL in config.py")
        return

    print("[Camera] Connected! Starting detection loop...")
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            # If video file ends, loop back to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1

        # ── Detect ────────────────────────────────────────────────────────────
        detections = detector.detect(frame)

        # ── Track ─────────────────────────────────────────────────────────────
        tracked     = tracker.update(detections)
        stopped_ids = tracker.get_stopped_vehicles()

        # ── Check all alarms ──────────────────────────────────────────────────
        alarms.check_congestion(len(detections))
        alarms.check_stopped_vehicles(stopped_ids, tracked)
        alarms.check_crash(detections)

        # ── Log vehicle count to DB every 30 frames ───────────────────────────
        if frame_count % 30 == 0:
            logger.log_vehicle_count(len(detections))

        # ── Draw annotations ──────────────────────────────────────────────────
        annotated, display_count = draw_detections(
            frame.copy(), detections, tracked, stopped_ids
        )

        # ── Push to browser dashboard ─────────────────────────────────────────
        update_state(
            frame         = annotated,
            vehicle_count = display_count,
            alarms        = alarms.get_status(),
            recent_events = [list(e) for e in logger.get_recent_events(10)]
        )

        # ── Local cv2 window (off by default) ─────────────────────────────────
        if SHOW_LOCAL_WINDOW:
            cv2.imshow("Traffic Management", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if SHOW_LOCAL_WINDOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()