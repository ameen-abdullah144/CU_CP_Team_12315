import sqlite3
import logging
import os
from datetime import datetime

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/events.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


class EventLogger:
    def __init__(self):
        self.db_path = "logs/traffic_log.db"
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                type      TEXT,
                message   TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_counts (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                count     INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def log_event(self, event_type, message):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.warning(f"[{event_type.upper()}] {message}")
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO events (timestamp, type, message) VALUES (?, ?, ?)",
            (ts, event_type, message)
        )
        conn.commit()
        conn.close()

    def log_vehicle_count(self, count):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO vehicle_counts (timestamp, count) VALUES (?, ?)",
            (ts, count)
        )
        conn.commit()
        conn.close()

    def get_recent_events(self, limit=20):
        conn   = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT timestamp, type, message FROM events ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return rows