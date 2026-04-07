import time
import threading
import numpy as np
from config import (CONGESTION_THRESHOLD, CRASH_OVERLAP_THRESHOLD,
                    ALARM_COOLDOWN_SECONDS, SOUND_ALARM_ENABLED)


class AlarmManager:
    def __init__(self, logger):
        self.logger        = logger
        self.last_alarm    = {}
        self.active_alarms = {}

        self.sound_ready = False
        if SOUND_ALARM_ENABLED:
            try:
                import pygame
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
                self.pygame = pygame
                self.sound_ready = True
                print("[Alarm] Sound system ready (pygame).")
            except Exception as e:
                print(f"[Alarm] Sound not available: {e}")

    def _can_trigger(self, alarm_type):
        return (time.time() - self.last_alarm.get(alarm_type, 0)) >= ALARM_COOLDOWN_SECONDS

    def _trigger(self, alarm_type, message, play_sound=True):
        if not self._can_trigger(alarm_type):
            return
        self.last_alarm[alarm_type]    = time.time()
        self.active_alarms[alarm_type] = True
        print(f"[ALARM] 🚨 {message}")
        self.logger.log_event(alarm_type, message)
        if play_sound and self.sound_ready:
            threading.Thread(target=self._beep, daemon=True).start()
        threading.Timer(
            ALARM_COOLDOWN_SECONDS,
            lambda: self.active_alarms.update({alarm_type: False})
        ).start()

    def _beep(self):
        """
        Generates a beep using pygame.
        Uses time.sleep (not pygame.time.wait) — safe in background thread.
        Plays 2 short beeps for urgency.
        """
        try:
            sample_rate = 44100
            duration    = 0.3
            freq        = 880

            t    = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            wave = np.ascontiguousarray(wave)

            sound = self.pygame.sndarray.make_sound(wave)

            sound.play()
            time.sleep(duration + 0.05)

            sound.play()
            time.sleep(duration + 0.05)

        except Exception as e:
            print(f"[Alarm] Beep error: {e}")

    def check_congestion(self, vehicle_count):
        if vehicle_count >= CONGESTION_THRESHOLD:
            self._trigger(
                "congestion",
                f"HIGH CONGESTION: {vehicle_count} vehicles detected on road!"
            )

    def check_stopped_vehicles(self, stopped_ids, tracked):
        for tid in stopped_ids:
            self._trigger(
                f"stopped_{tid}",
                f"STOPPED VEHICLE: Vehicle #{tid} has not moved for too long!"
            )

    def check_crash(self, detections):
        boxes = [d["box"] for d in detections]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou = self._compute_iou(boxes[i], boxes[j])
                if iou > CRASH_OVERLAP_THRESHOLD:
                    self._trigger(
                        "crash",
                        f"POSSIBLE CRASH/INCIDENT: Vehicles overlapping! IoU={iou:.2f}"
                    )

    def _compute_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        ix1   = max(x1, x2);       iy1 = max(y1, y2)
        ix2   = min(x1+w1, x2+w2); iy2 = min(y1+h1, y2+h2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        union = w1*h1 + w2*h2 - inter
        return inter / union if union > 0 else 0

    def get_status(self):
        return dict(self.active_alarms)