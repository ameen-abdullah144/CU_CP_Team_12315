import numpy as np
from config import STOPPED_FRAMES_LIMIT, STOPPED_MOVE_THRESHOLD


class VehicleTracker:
    """
    Centroid-based vehicle tracker with:
    - Larger match radius so fast vehicles don't get lost
    - Grace period before a track is deleted (reduces ID churn)
    - Label locked on first detection (no flickering class names)
    """

    MAX_DISAPPEARED = 10  # frames a track survives without a match

    def __init__(self):
        self.next_id     = 0
        self.tracked     = {}   # id -> track info
        self.disappeared = {}   # id -> frames since last seen

    def get_centroid(self, box):
        x, y, w, h = box
        return (int(x + w / 2), int(y + h / 2))

    def update(self, detections):
        new_centroids = [self.get_centroid(d["box"]) for d in detections]

        # ── First frame — register everything ─────────────────────────────────
        if not self.tracked:
            for i, det in enumerate(detections):
                self._register(new_centroids[i], det)
            return self.tracked

        old_ids       = list(self.tracked.keys())
        old_centroids = [self.tracked[i]["centroid"] for i in old_ids]

        matched_new = set()
        matched_old = set()

        # ── Match each new detection to nearest existing track ─────────────────
        for ni, nc in enumerate(new_centroids):
            best_dist = 150   # px — raised from 100 so fast vehicles still match
            best_oi   = None
            for oi, oc in enumerate(old_centroids):
                dist = np.linalg.norm(np.array(nc) - np.array(oc))
                if dist < best_dist:
                    best_dist = dist
                    best_oi   = oi

            if best_oi is not None and best_oi not in matched_old:
                tid   = old_ids[best_oi]
                old_c = self.tracked[tid]["centroid"]

                movement = np.linalg.norm(np.array(nc) - np.array(old_c))
                if movement < STOPPED_MOVE_THRESHOLD:
                    self.tracked[tid]["frames_stopped"] += 1
                else:
                    self.tracked[tid]["frames_stopped"] = 0

                self.tracked[tid]["centroid"]     = nc
                self.tracked[tid]["last_box"]     = detections[ni]["box"]
                self.tracked[tid]["frames_seen"] += 1
                self.disappeared[tid]             = 0

                matched_new.add(ni)
                matched_old.add(best_oi)

        # ── Register new detections that didn't match any track ────────────────
        for ni in range(len(new_centroids)):
            if ni not in matched_new:
                self._register(new_centroids[ni], detections[ni])

        # ── Give unmatched tracks a grace period before deleting ───────────────
        for oi, tid in enumerate(old_ids):
            if oi not in matched_old:
                self.disappeared[tid] = self.disappeared.get(tid, 0) + 1
                if self.disappeared[tid] > self.MAX_DISAPPEARED:
                    del self.tracked[tid]
                    if tid in self.disappeared:
                        del self.disappeared[tid]

        return self.tracked

    def _register(self, centroid, det):
        self.tracked[self.next_id] = {
            "centroid":       centroid,
            "frames_seen":    1,
            "frames_stopped": 0,
            "last_box":       det["box"],
            "label":          det["label"],    # locked — won't flicker
            "class_id":       det["class_id"]
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def get_stopped_vehicles(self):
        return [
            tid for tid, info in self.tracked.items()
            if info["frames_stopped"] >= STOPPED_FRAMES_LIMIT
        ]