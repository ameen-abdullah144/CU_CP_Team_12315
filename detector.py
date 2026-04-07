import cv2
import numpy as np
from config import (WEIGHTS_PATH, CONFIG_PATH, NAMES_PATH,
                    CONFIDENCE_THRESHOLD, NMS_THRESHOLD,
                    INPUT_SIZE, VEHICLE_CLASSES)


class VehicleDetector:
    def __init__(self):
        print("[Detector] Loading YOLOv4-tiny model...")
        self.net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)

        # CPU only
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        with open(NAMES_PATH, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.output_layers = self.net.getUnconnectedOutLayersNames()
        print("[Detector] Model loaded successfully.")

    def detect(self, frame):
        """
        Run detection on a frame.
        Returns list of dicts: {box, confidence, class_id, label}
        Only returns vehicle classes above confidence threshold.
        """
        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255, INPUT_SIZE,
            (0, 0, 0), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores     = detection[5:]
                class_id   = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if class_id not in VEHICLE_CLASSES:
                    continue
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w  = int(detection[2] * width)
                h  = int(detection[3] * height)
                x  = int(cx - w / 2)
                y  = int(cy - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

        # Non-max suppression — removes duplicate boxes on same vehicle
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD
        )

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    "box":        boxes[i],
                    "confidence": confidences[i],
                    "class_id":   class_ids[i],
                    "label":      self.classes[class_ids[i]]
                })

        return detections