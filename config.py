# ─── Camera Settings ───────────────────────────────────────────────────────────
CAMERA_URL = "test_video.mp4"   # swap to RTSP URL when real camera is ready

# ─── YOLO Settings ─────────────────────────────────────────────────────────────
WEIGHTS_PATH = "yolov4-tiny.weights"
CONFIG_PATH  = "yolov4-tiny.cfg"
NAMES_PATH   = "coco.names"

CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD        = 0.2
INPUT_SIZE           = (416, 416)

# ─── Vehicle Classes (COCO dataset IDs) ────────────────────────────────────────
# car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = [2, 3, 5, 7]

# ─── Alarm Thresholds ──────────────────────────────────────────────────────────
CONGESTION_THRESHOLD    = 10
STOPPED_FRAMES_LIMIT    = 180
STOPPED_MOVE_THRESHOLD  = 15
CRASH_OVERLAP_THRESHOLD = 0.4

# ─── Alarm Settings ────────────────────────────────────────────────────────────
ALARM_COOLDOWN_SECONDS = 15
SOUND_ALARM_ENABLED    = True

# ─── Dashboard Settings ────────────────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 5000
STREAM_FPS     = 15

# ─── Display Settings ──────────────────────────────────────────────────────────
SHOW_LOCAL_WINDOW = False   # True = also open cv2 popup window
SHOW_CONFIDENCE   = True    # show % on bounding box labels
MIN_DISPLAY_CONF  = 0.65    # don't draw boxes below this confidence